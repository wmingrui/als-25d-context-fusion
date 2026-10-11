from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    if os.name == "nt":
        raise ImportError(
            "Official Open3D Windows wheels do not include "
            "open3d.ml.torch."
        )
    from open3d.ml.torch.models import RandLANet
    RANDLANET_BACKEND = "open3d.ml.torch"
except Exception as open3d_import_error:
    from randlanet_torch_fallback import RandLANet
    RANDLANET_BACKEND = (
        "pure_torch_fallback "
        f"({type(open3d_import_error).__name__}: "
        f"{open3d_import_error})"
    )


NUM_CLASSES = 8
GATE_MODES = (
    "learned",
    "unit",
    "zero",
    "shuffled",
)

CLASS_NAMES = [
    "ground",
    "vegetation",
    "cars",
    "trucks",
    "power_lines",
    "fences",
    "poles",
    "buildings",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(base_seed, name):
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (base_seed + int(h[:8], 16)) % (2**32 - 1)


def clean_stem_from_deg_stem(stem, degradation_tag="deg_v1"):
    suffix = f"_{degradation_tag}"
    if not stem.endswith(suffix):
        raise ValueError(
            f"Expected degraded block stem to end with {suffix!r}, "
            f"got {stem!r}."
        )
    return stem[: -len(suffix)]


def load_npz(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def assert_matching_normalization(
    degraded_npz,
    restored_npz,
    degraded_path,
    restored_path,
):
    key = "normalization_stats_sha256"
    degraded_has = key in degraded_npz
    restored_has = key in restored_npz
    if not degraded_has and not restored_has:

        return
    if degraded_has != restored_has:
        raise RuntimeError(
            "Degraded/restored normalization metadata mismatch: "
            f"{degraded_path} vs {restored_path}."
        )
    degraded_hash = str(
        np.asarray(degraded_npz[key]).item()
    )
    restored_hash = str(
        np.asarray(restored_npz[key]).item()
    )
    if degraded_hash != restored_hash:
        raise RuntimeError(
            "Degraded and restored maps use different shared "
            "normalization statistics: "
            f"{degraded_hash} != {restored_hash}."
        )


def get_mask(npz, key, shape):
    if key in npz:
        return npz[key].astype(np.float32)
    return np.zeros(shape, dtype=np.float32)


def build_reliability_maps(deg_npz):
    """
    Returns:
        6,H,W:
          0 reliability
          1 valid
          2 hard
          3 low-density
          4 high-gradient
          5 roughness
    """
    maps = deg_npz["maps_norm"].astype(np.float32)
    _, H, W = maps.shape

    density = maps[4]
    height_gradient = maps[7]
    boundary = maps[8]

    valid = (
        deg_npz["valid_mask"].astype(np.float32)
        if "valid_mask" in deg_npz
        else np.ones((H, W), dtype=np.float32)
    )

    hard = get_mask(deg_npz, "structural_hard_mask", (H, W))
    low = get_mask(deg_npz, "low_density_mask", (H, W))
    high_gradient = get_mask(deg_npz, "high_gradient_mask", (H, W))
    rough = get_mask(deg_npz, "high_roughness_mask", (H, W))

    reliability = (
        0.45 * density
        + 0.20 * valid
        + 0.20 * (1.0 - boundary)
        + 0.15 * (1.0 - height_gradient)
    )

    penalty = (
        0.35 * hard
        + 0.25 * low
        + 0.20 * high_gradient
        + 0.15 * rough
    )

    reliability = reliability * (1.0 - np.clip(penalty, 0.0, 0.90))
    reliability = np.clip(reliability * valid, 0.0, 1.0)

    return np.stack(
        [
            reliability,
            valid,
            hard,
            low,
            high_gradient,
            rough,
        ],
        axis=0,
    ).astype(np.float32)


def sample_grid_at_points(points, maps, x0, y0, resolution):
    """
    points: N,3 in block coordinates
    maps: C,H,W
    """
    _, H, W = maps.shape

    ix = np.floor((points[:, 0] - x0) / resolution).astype(np.int32)
    iy = np.floor((points[:, 1] - y0) / resolution).astype(np.int32)

    ix = np.clip(ix, 0, W - 1)
    iy = np.clip(iy, 0, H - 1)

    return maps[:, iy, ix].T.astype(np.float32)


def sample_maps(points, map_npz):
    maps = map_npz["maps_norm"].astype(np.float32)
    x0 = float(map_npz["x0"])
    y0 = float(map_npz["y0"])
    resolution = float(map_npz["resolution"])

    return sample_grid_at_points(
        points,
        maps,
        x0,
        y0,
        resolution,
    )


def sample_reliability(points, deg_npz):
    maps = build_reliability_maps(deg_npz)
    x0 = float(deg_npz["x0"])
    y0 = float(deg_npz["y0"])
    resolution = float(deg_npz["resolution"])

    return sample_grid_at_points(
        points,
        maps,
        x0,
        y0,
        resolution,
    )


def knn_search(support, query, k):
    """
    support: Ns,3
    query:   Nq,3
    return:  Nq,k int64
    """
    if len(support) == 0:
        raise RuntimeError("Empty support point set.")

    k_eff = min(k, len(support))
    tree = cKDTree(support)

    _, idx = tree.query(query, k=k_eff)

    if k_eff == 1:
        idx = idx[:, None]

    if k_eff < k:
        pad = np.repeat(idx[:, -1:], k - k_eff, axis=1)
        idx = np.concatenate([idx, pad], axis=1)

    return idx.astype(np.int64)


def build_randlanet_hierarchy(
    coords,
    num_layers,
    num_neighbors,
    sub_sampling_ratio,
):
    """
    Reproduce the hierarchy expected by Open3D RandLANet.forward().
    """
    current = coords.astype(np.float32).copy()

    coords_list = []
    neighbor_list = []
    pool_list = []
    interp_list = []

    for layer in range(num_layers):
        neighbors = knn_search(
            support=current,
            query=current,
            k=num_neighbors,
        )

        n_sub = max(1, current.shape[0] // sub_sampling_ratio[layer])
        sub_points = current[:n_sub].copy()



        pool_idx = neighbors[:n_sub].copy()



        interp_idx = knn_search(
            support=sub_points,
            query=current,
            k=1,
        )

        coords_list.append(current)
        neighbor_list.append(neighbors)
        pool_list.append(pool_idx)
        interp_list.append(interp_idx)

        current = sub_points

    return {
        "coords": coords_list,
        "neighbor_indices": neighbor_list,
        "sub_idx": pool_list,
        "interp_idx": interp_list,
    }


class DALESRandLADataset(Dataset):
    def __init__(
        self,
        block_dir,
        clean_map_dir,
        deg_map_dir,
        restored_map_dir,
        split,
        mode,
        num_points,
        num_layers,
        num_neighbors,
        sub_sampling_ratio,
        degradation_tag="deg_v1",
        max_files=None,
        seed=42,
    ):
        self.block_dir = Path(block_dir) / split
        self.clean_map_dir = Path(clean_map_dir)
        self.deg_map_dir = Path(deg_map_dir)
        self.restored_map_dir = Path(restored_map_dir)

        self.split = split
        self.mode = mode
        self.num_points = num_points
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.sub_sampling_ratio = sub_sampling_ratio
        self.degradation_tag = str(degradation_tag)
        self.seed = seed

        self.files = sorted(
            self.block_dir.glob(
                f"*_{self.degradation_tag}.npz"
            )
        )

        if max_files is not None:
            self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise RuntimeError(
                "No block files found in "
                f"{self.block_dir} with degradation tag "
                f"{self.degradation_tag!r}."
            )

    def __len__(self):
        return len(self.files)

    def _load_prior_files(self, block_stem):
        clean_stem = clean_stem_from_deg_stem(
            block_stem,
            self.degradation_tag,
        )

        deg_fp = (
            self.deg_map_dir
            / self.split
            / f"{block_stem}_25d.npz"
        )

        restored_fp = (
            self.restored_map_dir
            / self.split
            / f"{clean_stem}_restored_unet_25d.npz"
        )

        deg_npz = load_npz(deg_fp)
        restored_npz = load_npz(restored_fp)
        assert_matching_normalization(
            deg_npz,
            restored_npz,
            deg_fp,
            restored_fp,
        )

        return deg_npz, restored_npz

    def __getitem__(self, idx):
        fp = self.files[idx]
        block = np.load(fp, allow_pickle=True)

        points = block["points"].astype(np.float32)
        points_local = block["points_local"].astype(np.float32)
        intensity = block["features"].astype(np.float32)
        labels = block["labels"].astype(np.int64)

        hard_all = (
            block["point_hard_mask"].astype(np.bool_)
            if "point_hard_mask" in block
            else np.zeros(len(points), dtype=np.bool_)
        )

        n = len(points)

        if self.split == "train":
            choice = np.random.choice(
                n,
                self.num_points,
                replace=n < self.num_points,
            )
        else:
            rng = np.random.default_rng(
                stable_seed(self.seed, fp.name)
            )
            choice = rng.choice(
                n,
                self.num_points,
                replace=n < self.num_points,
            )

        pts = points[choice]
        coords = points_local[choice]
        inten = intensity[choice]
        y = labels[choice]
        hard = hard_all[choice].astype(np.float32)


        coords = coords.astype(np.float32)


        xyz_feature = coords.copy()
        xyz_feature[:, 0] /= 25.0
        xyz_feature[:, 1] /= 25.0
        xyz_feature[:, 2] = np.clip(
            xyz_feature[:, 2] / 20.0,
            -2.0,
            2.0,
        )

        base = np.concatenate(
            [xyz_feature, inten],
            axis=1,
        ).astype(np.float32)

        prior_restored = np.zeros(
            (self.num_points, 9),
            dtype=np.float32,
        )
        prior_degraded = np.zeros_like(prior_restored)
        delta = np.zeros_like(prior_restored)

        rel = np.zeros(
            (self.num_points, 6),
            dtype=np.float32,
        )

        if self.mode != "base":
            deg_npz, restored_npz = self._load_prior_files(fp.stem)

            prior_degraded = sample_maps(pts, deg_npz)
            prior_restored = sample_maps(pts, restored_npz)
            delta = prior_restored - prior_degraded
            rel = sample_reliability(pts, deg_npz)

        hierarchy = build_randlanet_hierarchy(
            coords=coords,
            num_layers=self.num_layers,
            num_neighbors=self.num_neighbors,
            sub_sampling_ratio=self.sub_sampling_ratio,
        )

        return {
            "base": base,
            "restored": prior_restored,
            "degraded": prior_degraded,
            "delta": delta,
            "rel": rel,
            "labels": y,
            "hard": hard,
            "coords": hierarchy["coords"],
            "neighbor_indices": hierarchy["neighbor_indices"],
            "sub_idx": hierarchy["sub_idx"],
            "interp_idx": hierarchy["interp_idx"],
            "name": fp.name,
        }


def randla_collate(batch):
    out = {
        "base": torch.from_numpy(
            np.stack([x["base"] for x in batch], axis=0)
        ),
        "restored": torch.from_numpy(
            np.stack([x["restored"] for x in batch], axis=0)
        ),
        "degraded": torch.from_numpy(
            np.stack([x["degraded"] for x in batch], axis=0)
        ),
        "delta": torch.from_numpy(
            np.stack([x["delta"] for x in batch], axis=0)
        ),
        "rel": torch.from_numpy(
            np.stack([x["rel"] for x in batch], axis=0)
        ),
        "labels": torch.from_numpy(
            np.stack([x["labels"] for x in batch], axis=0)
        ).long(),
        "hard": torch.from_numpy(
            np.stack([x["hard"] for x in batch], axis=0)
        ).float(),
        "name": [x["name"] for x in batch],
    }

    num_layers = len(batch[0]["coords"])

    for key in [
        "coords",
        "neighbor_indices",
        "sub_idx",
        "interp_idx",
    ]:
        out[key] = []

        for layer in range(num_layers):
            arr = np.stack(
                [item[key][layer] for item in batch],
                axis=0,
            )

            tensor = torch.from_numpy(arr)

            if key != "coords":
                tensor = tensor.long()
            else:
                tensor = tensor.float()

            out[key].append(tensor)

    return out


class RGDIFusion(nn.Module):
    def __init__(
        self,
        base_ch=4,
        prior_ch=9,
        rel_ch=6,
        embed_ch=32,
        gate_bias=0.30,
        use_explicit_quality_modulation=True,
        use_reliability_descriptors=True,
        gate_mode="learned",
    ):
        super().__init__()
        self.gate_bias = gate_bias
        if gate_mode not in GATE_MODES:
            raise ValueError(
                f"Unknown gate_mode={gate_mode}. "
                f"Expected one of {GATE_MODES}."
            )
        self.gate_mode = gate_mode
        self.use_explicit_quality_modulation = bool(
            use_explicit_quality_modulation
        )
        self.use_reliability_descriptors = bool(
            use_reliability_descriptors
        )

        self.base_embed = nn.Sequential(
            nn.Conv1d(base_ch, embed_ch, 1, bias=False),
            nn.BatchNorm1d(embed_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(embed_ch, embed_ch, 1, bias=False),
            nn.BatchNorm1d(embed_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.degraded_embed = nn.Sequential(
            nn.Conv1d(prior_ch, embed_ch, 1, bias=False),
            nn.BatchNorm1d(embed_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(embed_ch, embed_ch, 1, bias=False),
            nn.BatchNorm1d(embed_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.delta_embed = nn.Sequential(
            nn.Conv1d(prior_ch, embed_ch, 1, bias=False),
            nn.BatchNorm1d(embed_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(embed_ch, embed_ch, 1, bias=False),
            nn.BatchNorm1d(embed_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.gate_net = nn.Sequential(
            nn.Conv1d(
                base_ch + prior_ch + prior_ch + rel_ch,
                embed_ch,
                1,
                bias=False,
            ),
            nn.BatchNorm1d(embed_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(embed_ch, embed_ch, 1),
        )

    def forward(self, base, degraded, delta, rel):

        base_t = base.transpose(1, 2).contiguous()
        degraded_t = degraded.transpose(1, 2).contiguous()
        delta_t = delta.transpose(1, 2).contiguous()
        rel_t = rel.transpose(1, 2).contiguous()





        if self.gate_mode == "shuffled":
            shift = max(1, delta_t.shape[-1] // 2)
            delta_used_t = torch.roll(
                delta_t,
                shifts=shift,
                dims=-1,
            )
        else:
            delta_used_t = delta_t

        base_feat = self.base_embed(base_t)
        degraded_feat = self.degraded_embed(degraded_t)
        correction_feat = self.delta_embed(delta_used_t)



        rel_for_gate = (
            rel_t
            if self.use_reliability_descriptors
            else torch.zeros_like(rel_t)
        )

        gate_input = torch.cat(
            [
                base_t,
                degraded_t,
                delta_used_t,
                rel_for_gate,
            ],
            dim=1,
        )

        learned_gate = torch.sigmoid(
            self.gate_net(gate_input)
        )

        reliability = rel_t[:, 0:1]
        hard = rel_t[:, 2:3]
        low = rel_t[:, 3:4]
        high_gradient = rel_t[:, 4:5]
        rough = rel_t[:, 5:6]

        unreliable = 1.0 - reliability

        explicit = (
            0.60 * unreliable
            + 0.15 * hard
            + 0.10 * low
            + 0.10 * high_gradient
            + 0.05 * rough
        ).clamp(0.0, 1.0)

        if (
            self.use_explicit_quality_modulation
            and self.use_reliability_descriptors
        ):
            explicit_gate = (
                self.gate_bias
                + (1.0 - self.gate_bias) * explicit
            )
        else:



            explicit_gate = torch.ones_like(explicit)

        if self.gate_mode == "unit":
            gate = torch.ones_like(learned_gate)
        elif self.gate_mode == "zero":
            gate = torch.zeros_like(learned_gate)
        else:
            gate = learned_gate * explicit_gate

        fused = (
            base_feat
            + degraded_feat
            + gate * correction_feat
        )

        fused = fused.transpose(1, 2).contiguous()


        return torch.cat([base, fused], dim=-1)


class InternalRGDIInjector(nn.Module):
    """Task-driven correction injection inside a RandLA encoder stage.

    The module operates on the feature map after a DilatedResidualBlock and
    before RandLA's random sampling. The structural correction remains aligned
    with the current point coordinates. A small learnable residual coefficient
    keeps the initial model close to the proven input-only RGDI baseline.
    """

    def __init__(
        self,
        feature_ch,
        prior_ch=9,
        rel_ch=6,
        hidden_ch=32,
        gate_bias=0.30,
        alpha_init=0.10,
        use_explicit_quality_modulation=True,
        use_reliability_descriptors=True,
        gate_mode="learned",
    ):
        super().__init__()

        self.feature_ch = int(feature_ch)
        self.gate_bias = float(gate_bias)
        if gate_mode not in GATE_MODES:
            raise ValueError(
                f"Unknown gate_mode={gate_mode}. "
                f"Expected one of {GATE_MODES}."
            )
        self.gate_mode = gate_mode
        self.use_explicit_quality_modulation = bool(
            use_explicit_quality_modulation
        )
        self.use_reliability_descriptors = bool(
            use_reliability_descriptors
        )

        self.feature_context = nn.Sequential(
            nn.Conv1d(
                self.feature_ch,
                hidden_ch,
                1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.prior_context = nn.Sequential(
            nn.Conv1d(
                prior_ch + prior_ch + rel_ch,
                hidden_ch,
                1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.gate_head = nn.Sequential(
            nn.Conv1d(
                hidden_ch * 2,
                hidden_ch,
                1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                hidden_ch,
                self.feature_ch,
                1,
            ),
        )

        self.delta_projection = nn.Sequential(
            nn.Conv1d(
                prior_ch,
                self.feature_ch,
                1,
                bias=False,
            ),
            nn.BatchNorm1d(self.feature_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )



        self.alpha_raw = nn.Parameter(
            torch.tensor(float(alpha_init))
        )

    def forward(
        self,
        feature,
        degraded,
        delta,
        rel,
    ):
        """Inject correction at the current RandLA resolution.

        Args:
            feature:  B,C,N,1 RandLA feature after an encoder block.
            degraded: B,N,9 degraded structural prior.
            delta:    B,N,9 restored-minus-degraded correction.
            rel:      B,N,6 existing structural reliability descriptors.
        """

        if feature.ndim != 4 or feature.shape[-1] != 1:
            raise RuntimeError(
                "InternalRGDIInjector expects feature shape B,C,N,1, "
                f"got {tuple(feature.shape)}"
            )

        feature_1d = feature.squeeze(-1)
        degraded_t = degraded.transpose(1, 2).contiguous()
        delta_t = delta.transpose(1, 2).contiguous()
        rel_t = rel.transpose(1, 2).contiguous()

        if self.gate_mode == "shuffled":
            shift = max(1, delta_t.shape[-1] // 2)
            delta_used_t = torch.roll(
                delta_t,
                shifts=shift,
                dims=-1,
            )
        else:
            delta_used_t = delta_t

        if feature_1d.shape[-1] != delta_used_t.shape[-1]:
            raise RuntimeError(
                "Internal RGDI point-count mismatch: "
                f"feature N={feature_1d.shape[-1]}, "
                f"delta N={delta_used_t.shape[-1]}"
            )

        semantic_context = self.feature_context(feature_1d)

        rel_for_gate = (
            rel_t
            if self.use_reliability_descriptors
            else torch.zeros_like(rel_t)
        )

        prior_context = self.prior_context(
            torch.cat(
                [degraded_t, delta_used_t, rel_for_gate],
                dim=1,
            )
        )

        learned_gate = torch.sigmoid(
            self.gate_head(
                torch.cat(
                    [semantic_context, prior_context],
                    dim=1,
                )
            )
        )

        reliability = rel_t[:, 0:1]
        hard = rel_t[:, 2:3]
        low = rel_t[:, 3:4]
        high_gradient = rel_t[:, 4:5]
        rough = rel_t[:, 5:6]

        unreliable = 1.0 - reliability

        explicit = (
            0.60 * unreliable
            + 0.15 * hard
            + 0.10 * low
            + 0.10 * high_gradient
            + 0.05 * rough
        ).clamp(0.0, 1.0)

        if (
            self.use_explicit_quality_modulation
            and self.use_reliability_descriptors
        ):
            explicit_gate = (
                self.gate_bias
                + (1.0 - self.gate_bias) * explicit
            )
        else:

            explicit_gate = torch.ones_like(explicit)

        if self.gate_mode == "unit":
            gate = torch.ones_like(learned_gate)
        elif self.gate_mode == "zero":
            gate = torch.zeros_like(learned_gate)
        else:
            gate = learned_gate * explicit_gate

        correction_feature = self.delta_projection(delta_used_t)
        residual_strength = torch.tanh(self.alpha_raw)

        updated = (
            feature_1d
            + residual_strength
            * gate
            * correction_feature
        )

        return updated.unsqueeze(-1)

    def effective_alpha(self):
        return float(
            torch.tanh(self.alpha_raw)
            .detach()
            .cpu()
            .item()
        )


class RandLASegmentationModel(nn.Module):
    def __init__(
        self,
        mode,
        num_points,
        num_neighbors,
        num_layers,
        sub_sampling_ratio,
        dim_output,
        dim_features,
        embed_ch,
        gate_bias,
        inject_stages,
        internal_hidden_ch,
        internal_alpha_init,
        use_explicit_quality_modulation,
        use_reliability_descriptors,
        gate_mode,
        dropout,
        device,
    ):
        super().__init__()

        self.mode = mode
        self.gate_mode = gate_mode
        self.inject_stages = int(inject_stages)
        self.num_layers = int(num_layers)
        self.use_explicit_quality_modulation = bool(
            use_explicit_quality_modulation
        )
        self.use_reliability_descriptors = bool(
            use_reliability_descriptors
        )

        if self.inject_stages < 0 or self.inject_stages > 2:
            raise ValueError(
                "inject_stages must be 0, 1, or 2. "
                f"Got {self.inject_stages}."
            )

        if self.inject_stages > self.num_layers:
            raise ValueError(
                "inject_stages cannot exceed num_layers: "
                f"{self.inject_stages} > {self.num_layers}"
            )

        if mode != "restored_rgdi" and self.inject_stages != 0:
            raise ValueError(
                "Internal injection is only defined for "
                "--mode restored_rgdi. Use --inject_stages 0 "
                f"for mode={mode}."
            )

        if mode == "base":
            in_channels = 4
            self.rgdi = None

        elif mode in {
            "degraded_concat",
            "restored_concat",
            "delta_concat",
        }:

            in_channels = 13
            self.rgdi = None

        elif mode == "restored_rgdi":
            in_channels = 4 + embed_ch

            self.rgdi = RGDIFusion(
                base_ch=4,
                prior_ch=9,
                rel_ch=6,
                embed_ch=embed_ch,
                gate_bias=gate_bias,
                use_explicit_quality_modulation=(
                    self.use_explicit_quality_modulation
                ),
                use_reliability_descriptors=(
                    self.use_reliability_descriptors
                ),
                gate_mode=self.gate_mode,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.backbone = RandLANet(
            num_neighbors=num_neighbors,
            num_layers=num_layers,
            num_points=num_points,
            num_classes=NUM_CLASSES,
            ignored_label_inds=[],
            sub_sampling_ratio=sub_sampling_ratio,
            in_channels=in_channels,
            dim_features=dim_features,
            dim_output=dim_output,
            grid_size=0.5,
            dropout=dropout,
            device=str(device),
        )




        stage_channels = [2 * int(ch) for ch in dim_output]

        self.internal_injectors = nn.ModuleList()

        if mode == "restored_rgdi":
            for stage_idx in range(self.inject_stages):
                self.internal_injectors.append(
                    InternalRGDIInjector(
                        feature_ch=stage_channels[stage_idx],
                        prior_ch=9,
                        rel_ch=6,
                        hidden_ch=internal_hidden_ch,
                        gate_bias=gate_bias,
                        alpha_init=internal_alpha_init,
                        use_explicit_quality_modulation=(
                            self.use_explicit_quality_modulation
                        ),
                        use_reliability_descriptors=(
                            self.use_reliability_descriptors
                        ),
                        gate_mode=self.gate_mode,
                    )
                )


        self.backbone.device = device

    def _build_input_features(self, batch):
        base = batch["base"]

        if self.mode == "base":
            return base

        if self.mode == "degraded_concat":
            return torch.cat(
                [base, batch["degraded"]],
                dim=-1,
            )

        if self.mode == "restored_concat":
            return torch.cat(
                [base, batch["restored"]],
                dim=-1,
            )

        if self.mode == "delta_concat":
            return torch.cat(
                [base, batch["delta"]],
                dim=-1,
            )

        return self.rgdi(
            base,
            batch["degraded"],
            batch["delta"],
            batch["rel"],
        )

    def _forward_multiscale_rgdi(self, features, batch):
        """Replicate Open3D RandLANet.forward with internal RGDI hooks."""

        backbone = self.backbone
        cfg = backbone.cfg

        feat = features.to(backbone.device)
        coords_list = [
            arr.to(backbone.device)
            for arr in batch["coords"]
        ]
        neighbor_indices_list = [
            arr.to(backbone.device)
            for arr in batch["neighbor_indices"]
        ]
        subsample_indices_list = [
            arr.to(backbone.device)
            for arr in batch["sub_idx"]
        ]
        interpolation_indices_list = [
            arr.to(backbone.device)
            for arr in batch["interp_idx"]
        ]

        current_degraded = batch["degraded"]
        current_delta = batch["delta"]
        current_rel = batch["rel"]

        feat = (
            backbone.fc0(feat)
            .transpose(-2, -1)
            .unsqueeze(-1)
        )
        feat = backbone.bn0(feat)
        feat = F.leaky_relu(feat, negative_slope=0.2)

        encoder_feat_list = []

        for i in range(cfg.num_layers):
            feat_encoder_i = backbone.encoder[i](
                coords_list[i],
                feat,
                neighbor_indices_list[i],
            )

            if i < self.inject_stages:
                expected_ch = (
                    self.internal_injectors[i]
                    .feature_ch
                )

                if feat_encoder_i.shape[1] != expected_ch:
                    raise RuntimeError(
                        "Unexpected RandLA encoder channel count at "
                        f"stage {i}: expected {expected_ch}, "
                        f"got {feat_encoder_i.shape[1]}. "
                        "Check Open3D RandLANet implementation."
                    )

                feat_encoder_i = self.internal_injectors[i](
                    feat_encoder_i,
                    current_degraded,
                    current_delta,
                    current_rel,
                )

            feat_sampled_i = backbone.random_sample(
                feat_encoder_i,
                subsample_indices_list[i],
            )

            if i == 0:
                encoder_feat_list.append(
                    feat_encoder_i.clone()
                )

            encoder_feat_list.append(
                feat_sampled_i.clone()
            )
            feat = feat_sampled_i




            next_n = feat_sampled_i.shape[2]
            current_degraded = current_degraded[
                :, :next_n, :
            ]
            current_delta = current_delta[
                :, :next_n, :
            ]
            current_rel = current_rel[
                :, :next_n, :
            ]

        feat = backbone.mlp(feat)

        for i in range(cfg.num_layers):
            feat_interpolation_i = (
                backbone.nearest_interpolation(
                    feat,
                    interpolation_indices_list[-i - 1],
                )
            )

            feat_decoder_i = torch.cat(
                [
                    encoder_feat_list[-i - 2],
                    feat_interpolation_i,
                ],
                dim=1,
            )

            feat_decoder_i = backbone.decoder[i](
                feat_decoder_i
            )
            feat = feat_decoder_i

        scores = backbone.fc1(feat)
        return scores.squeeze(3).transpose(1, 2)

    def internal_alpha_values(self):
        return [
            module.effective_alpha()
            for module in self.internal_injectors
        ]

    def forward(self, batch):
        features = self._build_input_features(batch)

        if (
            self.mode == "restored_rgdi"
            and self.inject_stages > 0
        ):
            logits = self._forward_multiscale_rgdi(
                features,
                batch,
            )
        else:


            randla_inputs = {
                "features": features,
                "coords": batch["coords"],
                "neighbor_indices": batch[
                    "neighbor_indices"
                ],
                "sub_idx": batch["sub_idx"],
                "interp_idx": batch["interp_idx"],
            }
            logits = self.backbone(randla_inputs)

        if logits.ndim != 3:
            raise RuntimeError(
                "Unexpected RandLANet output shape: "
                f"{logits.shape}"
            )

        if logits.shape[-1] != NUM_CLASSES:
            raise RuntimeError(
                f"Expected last dimension={NUM_CLASSES}, "
                f"got {logits.shape}"
            )

        return logits

def compute_class_weights(block_dir):
    meta_fp = Path(block_dir) / "degradation_metadata.csv"

    if not meta_fp.exists():
        print("[WARN] degradation_metadata.csv not found.")
        return None

    df = pd.read_csv(meta_fp)
    df = df[df["split"] == "train"]

    counts = []

    for class_id in range(NUM_CLASSES):
        prefix = f"class_{class_id}_"
        cols = [
            c for c in df.columns
            if c.startswith(prefix)
        ]

        if not cols:
            counts.append(1.0)
        else:
            counts.append(
                float(df[cols[0]].sum()) + 1.0
            )

    counts = np.asarray(counts, dtype=np.float64)
    freq = counts / counts.sum()

    weights = 1.0 / np.sqrt(freq + 1e-12)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 8.0)

    print("[CLASS COUNTS]", counts.astype(np.int64))
    print("[CLASS WEIGHTS]", weights)

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def update_confusion(confusion, pred, target):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    valid = (
        (target >= 0)
        & (target < NUM_CLASSES)
    )

    pred = pred[valid]
    target = target[valid]

    values = target * NUM_CLASSES + pred

    bincount = np.bincount(
        values,
        minlength=NUM_CLASSES * NUM_CLASSES,
    )

    confusion += bincount.reshape(
        NUM_CLASSES,
        NUM_CLASSES,
    )


def metrics_from_confusion(confusion):
    tp = np.diag(confusion).astype(np.float64)
    gt = confusion.sum(axis=1).astype(np.float64)
    pred = confusion.sum(axis=0).astype(np.float64)
    union = gt + pred - tp

    iou = tp / np.maximum(union, 1.0)
    acc = tp / np.maximum(gt, 1.0)

    oa = tp.sum() / max(confusion.sum(), 1.0)

    return {
        "oa": float(oa),
        "mIoU": float(np.mean(iou)),
        "mAcc": float(np.mean(acc)),
        "per_class_iou": iou.tolist(),
        "per_class_acc": acc.tolist(),
    }


def move_batch_to_device(batch, device):
    for key in [
        "base",
        "restored",
        "degraded",
        "delta",
        "rel",
        "labels",
        "hard",
    ]:
        batch[key] = batch[key].to(
            device,
            non_blocking=True,
        )

    for key in [
        "coords",
        "neighbor_indices",
        "sub_idx",
        "interp_idx",
    ]:
        batch[key] = [
            tensor.to(device, non_blocking=True)
            for tensor in batch[key]
        ]

    return batch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    hard_confusion = np.zeros_like(confusion)

    loss_sum = 0.0
    sample_count = 0

    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_batch_to_device(batch, device)

        logits = model(batch)
        labels = batch["labels"]

        loss = F.cross_entropy(
            logits.reshape(-1, NUM_CLASSES),
            labels.reshape(-1),
        )

        pred = logits.argmax(dim=-1)

        pred_np = pred.cpu().numpy()
        labels_np = labels.cpu().numpy()
        hard_np = (
            batch["hard"].cpu().numpy() > 0.5
        )

        update_confusion(
            confusion,
            pred_np,
            labels_np,
        )

        if hard_np.any():
            update_confusion(
                hard_confusion,
                pred_np[hard_np],
                labels_np[hard_np],
            )

        batch_size = labels.shape[0]
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size

    normal = metrics_from_confusion(confusion)
    hard = metrics_from_confusion(hard_confusion)

    return {
        "loss": loss_sum / max(sample_count, 1),
        "oa": normal["oa"],
        "mIoU": normal["mIoU"],
        "mAcc": normal["mAcc"],
        "hard_oa": hard["oa"],
        "hard_mIoU": hard["mIoU"],
        "hard_mAcc": hard["mAcc"],
        "per_class_iou": normal["per_class_iou"],
        "hard_per_class_iou": hard["per_class_iou"],
    }


def train(args):
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("[INFO] device:", device)
    print("[INFO] RandLA-Net backend:", RANDLANET_BACKEND)
    print("[INFO] mode:", args.mode)
    print("[INFO] gate mode:", args.gate_mode)
    print("[INFO] degradation tag:", args.degradation_tag)
    print(
        "[INFO] explicit quality modulation:",
        not args.disable_explicit_quality_modulation,
    )
    print(
        "[INFO] learned reliability descriptors:",
        not args.disable_reliability_descriptors,
    )

    if args.disable_reliability_descriptors:
        reliability_mode = "none"
    elif args.disable_explicit_quality_modulation:
        reliability_mode = "learned_only"
    else:
        reliability_mode = "full"

    print("[INFO] reliability ablation mode:", reliability_mode)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    sub_sampling_ratio = [
        int(x)
        for x in args.sub_sampling_ratio.split(",")
    ]

    dim_output = [
        int(x)
        for x in args.dim_output.split(",")
    ]

    if len(sub_sampling_ratio) != args.num_layers:
        raise ValueError(
            "sub_sampling_ratio length must equal num_layers"
        )

    if len(dim_output) != args.num_layers:
        raise ValueError(
            "dim_output length must equal num_layers"
        )

    if (
        args.disable_explicit_quality_modulation
        and args.mode != "restored_rgdi"
    ):
        raise ValueError(
            "--disable_explicit_quality_modulation is only meaningful "
            "with --mode restored_rgdi."
        )

    if (
        args.disable_reliability_descriptors
        and args.mode != "restored_rgdi"
    ):
        raise ValueError(
            "--disable_reliability_descriptors is only meaningful "
            "with --mode restored_rgdi."
        )

    if (
        args.disable_reliability_descriptors
        and not args.disable_explicit_quality_modulation
    ):
        raise ValueError(
            "The strict no-reliability control must also remove the "
            "explicit reliability multiplier. Add "
            "--disable_explicit_quality_modulation."
        )

    if args.mode != "restored_rgdi" and args.gate_mode != "learned":
        raise ValueError(
            "--gate_mode only changes --mode restored_rgdi. "
            f"Use --gate_mode learned for mode={args.mode}."
        )

    if args.eval_only_checkpoint and args.skip_test:
        raise ValueError(
            "--skip_test is a training-only safeguard and cannot be "
            "combined with --eval_only_checkpoint. Select evaluation "
            "splits with --eval_only_splits instead."
        )

    eval_splits = {
        item.strip()
        for item in args.eval_only_splits.split(",")
        if item.strip()
    }
    invalid_eval_splits = eval_splits - {"val", "test"}
    if not eval_splits or invalid_eval_splits:
        raise ValueError(
            "--eval_only_splits must be a non-empty comma-separated "
            "subset of {val,test}; got "
            f"{args.eval_only_splits!r}."
        )

    dataset_kwargs = dict(
        block_dir=args.block_dir,
        clean_map_dir=args.clean_map_dir,
        deg_map_dir=args.deg_map_dir,
        restored_map_dir=args.restored_map_dir,
        mode=args.mode,
        num_points=args.num_points,
        num_layers=args.num_layers,
        num_neighbors=args.num_neighbors,
        sub_sampling_ratio=sub_sampling_ratio,
        degradation_tag=args.degradation_tag,
        seed=args.seed,
    )

    train_set = DALESRandLADataset(
        split="train",
        max_files=args.max_train_files,
        **dataset_kwargs,
    )

    val_set = DALESRandLADataset(
        split="val",
        max_files=args.max_val_files,
        **dataset_kwargs,
    )

    need_test_loader = (
        not args.skip_test
        and (
            not args.eval_only_checkpoint
            or "test" in eval_splits
        )
    )
    test_set = None
    if need_test_loader:
        test_set = DALESRandLADataset(
            split="test",
            max_files=args.max_test_files,
            **dataset_kwargs,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=randla_collate,
        persistent_workers=args.workers > 0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=randla_collate,
        persistent_workers=args.workers > 0,
    )

    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=randla_collate,
            persistent_workers=args.workers > 0,
        )

    model = RandLASegmentationModel(
        mode=args.mode,
        num_points=args.num_points,
        num_neighbors=args.num_neighbors,
        num_layers=args.num_layers,
        sub_sampling_ratio=sub_sampling_ratio,
        dim_output=dim_output,
        dim_features=args.dim_features,
        embed_ch=args.embed_ch,
        gate_bias=args.gate_bias,
        inject_stages=args.inject_stages,
        internal_hidden_ch=args.internal_hidden_ch,
        internal_alpha_init=args.internal_alpha_init,
        use_explicit_quality_modulation=(
            not args.disable_explicit_quality_modulation
        ),
        use_reliability_descriptors=(
            not args.disable_reliability_descriptors
        ),
        gate_mode=args.gate_mode,
        dropout=args.dropout,
        device=device,
    ).to(device)

    model_summary = {
        "total_parameters": int(
            sum(p.numel() for p in model.parameters())
        ),
        "trainable_parameters": int(
            sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )
        ),
        "backbone_parameters": int(
            sum(
                p.numel()
                for p in model.backbone.parameters()
            )
        ),
        "input_rgdi_parameters": int(
            sum(
                p.numel()
                for p in model.rgdi.parameters()
            )
            if model.rgdi is not None
            else 0
        ),
        "internal_rgdi_parameters": int(
            sum(
                p.numel()
                for p in model.internal_injectors.parameters()
            )
        ),
        "mode": args.mode,
        "gate_mode": args.gate_mode,
        "inject_stages": int(args.inject_stages),
    }
    with open(
        out_dir / "model_summary.json",
        "w",
    ) as f:
        json.dump(model_summary, f, indent=2)
    print(
        "[INFO] parameters:",
        model_summary["total_parameters"],
    )

    if args.eval_only_checkpoint:
        eval_fp = Path(args.eval_only_checkpoint)
        if not eval_fp.exists():
            raise FileNotFoundError(eval_fp)

        print("[EVAL ONLY] loading:", eval_fp)
        eval_ckpt = torch.load(
            eval_fp,
            map_location="cpu",
        )

        model.load_state_dict(
            eval_ckpt["model"],
            strict=True,
        )
        model.to(device)
        model.backbone.device = device

        if "val" in eval_splits:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
            )
            with open(
                out_dir / "eval_only_val_metrics.json",
                "w",
            ) as f:
                json.dump(val_metrics, f, indent=2)
            print("[EVAL ONLY VAL]")
            print(json.dumps(val_metrics, indent=2))

        if "test" in eval_splits:
            if test_loader is None:
                raise RuntimeError(
                    "Test evaluation was requested but no test loader "
                    "was constructed."
                )
            test_metrics = evaluate(
                model,
                test_loader,
                device,
            )
            with open(
                out_dir / "test_metrics.json",
                "w",
            ) as f:
                json.dump(test_metrics, f, indent=2)
            print("[EVAL ONLY TEST]")
            print(json.dumps(test_metrics, indent=2))

        print("[DONE]")
        return

    class_weights = compute_class_weights(
        args.block_dir
    )

    if (
        class_weights is not None
        and args.use_class_weight
    ):
        class_weights = class_weights.to(device)
    else:
        class_weights = None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and device.type == "cuda"
    )

    best_val_miou = -1.0
    best_val_epoch = None
    history = []
    start_epoch = 1

    if args.resume:
        resume_fp = Path(args.resume)

        if not resume_fp.exists():
            raise FileNotFoundError(resume_fp)

        print("[RESUME] loading:", resume_fp)

        resume_ckpt = torch.load(
            resume_fp,
            map_location="cpu",
        )

        model.load_state_dict(resume_ckpt["model"])
        model.to(device)
        model.backbone.device = device

        resume_epoch = int(
            resume_ckpt.get("epoch", 0)
        )
        start_epoch = resume_epoch + 1


        history_fp = out_dir / "history.csv"

        if history_fp.exists():
            history_df = pd.read_csv(history_fp)

            if "epoch" in history_df.columns:
                history_df = history_df[
                    history_df["epoch"] <= resume_epoch
                ].copy()

            history = history_df.to_dict("records")


        best_metrics_fp = (
            out_dir / "best_val_metrics.json"
        )

        if best_metrics_fp.exists():
            with open(best_metrics_fp) as f:
                best_metrics_old = json.load(f)

            best_val_miou = float(
                best_metrics_old.get("mIoU", -1.0)
            )
            if "epoch" in best_metrics_old:
                best_val_epoch = int(
                    best_metrics_old["epoch"]
                )
        elif history:
            best_row = max(
                history,
                key=lambda row: float(
                    row.get("val_mIoU", -1.0)
                ),
            )
            best_val_miou = float(
                best_row.get("val_mIoU", -1.0)
            )
            best_val_epoch = int(
                best_row.get("epoch", resume_epoch)
            )
        else:
            best_val_miou = float(
                resume_ckpt
                .get("val_metrics", {})
                .get("mIoU", -1.0)
            )
            best_val_epoch = resume_epoch

        has_training_state = all(
            key in resume_ckpt
            for key in [
                "optimizer",
                "scheduler",
                "scaler",
            ]
        )

        if (
            has_training_state
            and not args.reset_optimizer
        ):
            optimizer.load_state_dict(
                resume_ckpt["optimizer"]
            )
            scheduler.load_state_dict(
                resume_ckpt["scheduler"]
            )
            scaler.load_state_dict(
                resume_ckpt["scaler"]
            )

            print(
                "[RESUME] optimizer, scheduler and "
                "scaler restored"
            )

        else:


            remaining_epochs = max(
                1,
                args.epochs - resume_epoch,
            )

            scheduler = (
                torch.optim.lr_scheduler
                .CosineAnnealingLR(
                    optimizer,
                    T_max=remaining_epochs,
                    eta_min=args.lr * 0.05,
                )
            )

            print(
                "[RESUME] weight-only continuation; "
                "new optimizer and scheduler"
            )
            print(
                "[RESUME] new learning rate:",
                optimizer.param_groups[0]["lr"],
            )

        print(
            "[RESUME] checkpoint epoch:",
            resume_epoch,
        )
        print(
            "[RESUME] next epoch:",
            start_epoch,
        )
        print(
            "[RESUME] historical best val mIoU:",
            best_val_miou,
        )

    print("[INFO] train blocks:", len(train_set))
    print("[INFO] val blocks:", len(val_set))
    if test_set is None:
        print("[INFO] test blocks: not loaded")
    else:
        print("[INFO] test blocks:", len(test_set))
    print("[INFO] num_points:", args.num_points)
    print("[INFO] batch_size:", args.batch_size)
    print("[INFO] inject_stages:", args.inject_stages)
    print(
        "[INFO] initial internal alphas:",
        model.internal_alpha_values(),
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()

        running_loss = 0.0
        running_samples = 0

        pbar = tqdm(
            train_loader,
            desc=(
                f"{args.mode}_s{args.inject_stages} "
                f"{epoch}/{args.epochs}"
            ),
        )

        for batch in pbar:
            batch = move_batch_to_device(
                batch,
                device,
            )

            labels = batch["labels"]

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=args.amp
                and device.type == "cuda"
            ):
                logits = model(batch)

                loss = F.cross_entropy(
                    logits.reshape(-1, NUM_CLASSES),
                    labels.reshape(-1),
                    weight=class_weights,
                )

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.shape[0]
            running_loss += (
                float(loss.item()) * batch_size
            )
            running_samples += batch_size

            pbar.set_postfix(
                loss=running_loss
                / max(running_samples, 1)
            )

        scheduler.step()

        train_loss = (
            running_loss
            / max(running_samples, 1)
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        alpha_values = model.internal_alpha_values()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.time() - start_time,
            "train_loss": train_loss,
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
                if not isinstance(v, list)
            },
            **{
                f"internal_alpha_stage_{i}": value
                for i, value in enumerate(alpha_values)
            },
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            out_dir / "history.csv",
            index=False,
        )

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.6f} "
            f"val_mIoU={val_metrics['mIoU']:.6f} "
            f"val_OA={val_metrics['oa']:.6f} "
            f"val_hard_mIoU={val_metrics['hard_mIoU']:.6f} "
            f"val_hard_OA={val_metrics['hard_oa']:.6f} "
            f"internal_alpha={alpha_values}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_metrics": val_metrics,
            "inject_stages": args.inject_stages,
            "internal_alpha_values": alpha_values,
        }

        torch.save(
            checkpoint,
            out_dir / "last.pt",
        )

        if val_metrics["mIoU"] > best_val_miou:
            best_val_miou = val_metrics["mIoU"]
            best_val_epoch = epoch

            torch.save(
                checkpoint,
                out_dir / "best.pt",
            )

            with open(
                out_dir / "best_val_metrics.json",
                "w",
            ) as f:
                json.dump(
                    {
                        "epoch": epoch,
                        **val_metrics,
                    },
                    f,
                    indent=2,
                )

            print(
                f"[BEST] epoch={epoch}, "
                f"val_mIoU={best_val_miou:.6f}"
            )

    training_complete = {
        "status": "complete",
        "epochs_requested": int(args.epochs),
        "last_epoch": int(args.epochs),
        "best_val_epoch": best_val_epoch,
        "best_val_mIoU": float(best_val_miou),
        "mode": args.mode,
        "gate_mode": args.gate_mode,
        "inject_stages": int(args.inject_stages),
        "degradation_tag": args.degradation_tag,
        "test_skipped": bool(args.skip_test),
        "test_evaluated": False,
    }

    if args.skip_test:
        with open(
            out_dir / "training_complete.json",
            "w",
        ) as f:
            json.dump(training_complete, f, indent=2)
        print(
            "[DONE] validation-only training complete; "
            "test split was not loaded or evaluated."
        )
        return

    if test_loader is None:
        raise RuntimeError(
            "Internal error: final test evaluation requires a test loader."
        )

    print("[TEST] loading best checkpoint")

    checkpoint = torch.load(
        out_dir / "best.pt",
        map_location="cpu",
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.backbone.device = device

    test_metrics = evaluate(
        model,
        test_loader,
        device,
    )

    with open(
        out_dir / "test_metrics.json",
        "w",
    ) as f:
        json.dump(
            test_metrics,
            f,
            indent=2,
        )

    training_complete["test_evaluated"] = True
    with open(
        out_dir / "training_complete.json",
        "w",
    ) as f:
        json.dump(training_complete, f, indent=2)

    print("[TEST METRICS]")
    print(json.dumps(test_metrics, indent=2))
    print("[DONE]")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unified RandLA-Net mainline ablations for structural "
            "prior concatenation, input RGDI, and multi-scale RGDI."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "base",
            "degraded_concat",
            "restored_concat",
            "delta_concat",
            "restored_rgdi",
        ],
    )

    parser.add_argument(
        "--block_dir",
        type=str,
        default=(
            "/root/autodl-tmp/ALS-DiffBridge/"
            "data/processed/dales_objects/"
            "degraded_blocks_b50_v1"
        ),
    )
    parser.add_argument(
        "--degradation_tag",
        type=str,
        default="deg_v1",
        help=(
            "Filename tag after the clean block stem. For example, "
            "deg_v1 loads *_deg_v1.npz and deg_v3_mild loads "
            "*_deg_v3_mild.npz."
        ),
    )

    parser.add_argument(
        "--clean_map_dir",
        type=str,
        default=(
            "/root/autodl-tmp/ALS-DiffBridge/"
            "data/processed/dales_objects/"
            "maps_25d_b50_r050"
        ),
    )

    parser.add_argument(
        "--deg_map_dir",
        type=str,
        default=(
            "/root/autodl-tmp/ALS-DiffBridge/"
            "data/processed/dales_objects/"
            "maps_25d_b50_r050_degraded_v1"
        ),
    )

    parser.add_argument(
        "--restored_map_dir",
        type=str,
        default=(
            "/root/autodl-tmp/ALS-DiffBridge/"
            "outputs/restored_unet_b50_r050_v1_all/"
            "restored_maps"
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--num_neighbors", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)

    parser.add_argument(
        "--sub_sampling_ratio",
        type=str,
        default="4,4,4,4",
    )

    parser.add_argument(
        "--dim_output",
        type=str,
        default="16,64,128,256",
    )

    parser.add_argument("--dim_features", type=int, default=8)
    parser.add_argument("--embed_ch", type=int, default=32)
    parser.add_argument("--gate_bias", type=float, default=0.30)
    parser.add_argument(
        "--gate_mode",
        type=str,
        choices=GATE_MODES,
        default="learned",
        help=(
            "Correction control for restored_rgdi: learned is OA-GDI; "
            "unit applies the correction without a gate; zero removes "
            "the correction while preserving the branch parameters; "
            "shuffled destroys point-wise correction alignment using a "
            "deterministic cyclic shift."
        ),
    )
    parser.add_argument(
        "--disable_explicit_quality_modulation",
        action="store_true",
        help=(
            "Remove the hand-crafted explicit quality multiplier from "
            "both input and internal RGDI gates. The learned gates still "
            "receive the six reliability descriptors, so this isolates "
            "the value of explicit multiplicative modulation."
        ),
    )
    parser.add_argument(
        "--disable_reliability_descriptors",
        action="store_true",
        help=(
            "Strict reliability ablation: replace the six descriptors "
            "with zeros in both input and internal learned gates while "
            "preserving architecture and parameter count. Must be used "
            "together with --disable_explicit_quality_modulation."
        ),
    )
    parser.add_argument(
        "--inject_stages",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help=(
            "Number of earliest RandLA encoder stages receiving "
            "internal RGDI. 0=input-only, 1=encoder stage 0, "
            "2=encoder stages 0 and 1."
        ),
    )
    parser.add_argument(
        "--internal_hidden_ch",
        type=int,
        default=32,
        help="Hidden width of each internal RGDI gate.",
    )
    parser.add_argument(
        "--internal_alpha_init",
        type=float,
        default=0.10,
        help=(
            "Initial raw residual coefficient. The effective "
            "coefficient is tanh(value)."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--eval_only_checkpoint",
        type=str,
        default="",
        help=(
            "Load a checkpoint and evaluate val/test without "
            "training. For compatibility checks, use the same "
            "mode and inject_stages as the checkpoint."
        ),
    )
    parser.add_argument(
        "--eval_only_splits",
        type=str,
        default="val,test",
        help=(
            "Comma-separated splits evaluated with "
            "--eval_only_checkpoint. Use val or test to prevent "
            "accidental access to the other split."
        ),
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help=(
            "Training safeguard: do not construct or evaluate the test "
            "dataset after training. Use this for all model selection "
            "and ablation runs."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint path used to continue training.",
    )
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="Load model weights but initialize a new optimizer and scheduler.",
    )

    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
