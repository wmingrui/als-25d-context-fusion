from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


RAW_CHANNELS = [
    "height_max",
    "height_mean",
    "height_min",
    "height_std",
    "density",
    "log_density",
    "intensity_mean",
    "height_gradient",
    "boundary_response",
]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_name_from_deg(deg_name: str, degradation_tag: str):
    suffix = f"_{degradation_tag}_25d.npz"
    if not deg_name.endswith(suffix):
        raise ValueError(
            f"Expected map name ending in {suffix!r}, got "
            f"{deg_name!r}."
        )
    return f"{deg_name[:-len(suffix)]}_25d.npz"


def assert_matching_normalization(clean, degraded, clean_fp, deg_fp):
    key = "normalization_stats_sha256"
    clean_has = key in clean
    degraded_has = key in degraded
    if not clean_has and not degraded_has:
        return
    if clean_has != degraded_has:
        raise RuntimeError(
            "Clean/degraded normalization metadata mismatch: "
            f"{clean_fp} vs {deg_fp}."
        )
    clean_hash = str(np.asarray(clean[key]).item())
    degraded_hash = str(
        np.asarray(degraded[key]).item()
    )
    if clean_hash != degraded_hash:
        raise RuntimeError(
            "Clean and degraded maps use different shared "
            "normalization statistics."
        )


class ALS25DRestorationDataset(Dataset):
    def __init__(
        self,
        clean_dir,
        deg_dir,
        split,
        max_files=None,
        input_masks=True,
        degradation_tag="deg_v1",
    ):
        self.clean_dir = Path(clean_dir) / split
        self.deg_dir = Path(deg_dir) / split
        self.split = split
        self.input_masks = input_masks
        self.degradation_tag = degradation_tag

        self.deg_files = sorted(
            self.deg_dir.glob(
                f"*_{self.degradation_tag}_25d.npz"
            )
        )
        if max_files is not None:
            self.deg_files = self.deg_files[:max_files]

        if len(self.deg_files) == 0:
            raise RuntimeError(f"No degraded maps found in {self.deg_dir}")

    def __len__(self):
        return len(self.deg_files)

    def __getitem__(self, idx):
        deg_fp = self.deg_files[idx]
        clean_fp = self.clean_dir / clean_name_from_deg(
            deg_fp.name,
            self.degradation_tag,
        )

        if not clean_fp.exists():
            raise FileNotFoundError(f"Missing clean map: {clean_fp}")

        deg = np.load(deg_fp, allow_pickle=True)
        clean = np.load(clean_fp, allow_pickle=True)
        assert_matching_normalization(
            clean,
            deg,
            clean_fp,
            deg_fp,
        )

        deg_maps = deg["maps_norm"].astype(np.float32)
        clean_maps = clean["maps_norm"].astype(np.float32)

        deg_valid = deg["valid_mask"].astype(np.float32)[None]
        clean_valid = clean["valid_mask"].astype(np.float32)[None]

        deg_hard = deg["structural_hard_mask"].astype(np.float32)[None]
        deg_low = deg["low_density_mask"].astype(np.float32)[None]
        deg_grad = deg["high_gradient_mask"].astype(np.float32)[None]
        deg_rough = deg["high_roughness_mask"].astype(np.float32)[None]

        clean_hard = clean["structural_hard_mask"].astype(np.float32)[None]

        if self.input_masks:
            x = np.concatenate(
                [deg_maps, deg_valid, deg_hard, deg_low, deg_grad, deg_rough],
                axis=0,
            )
        else:
            x = deg_maps

        y = clean_maps

        sample = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "clean_valid": torch.from_numpy(clean_valid),
            "clean_hard": torch.from_numpy(clean_hard),
            "deg_valid": torch.from_numpy(deg_valid),
            "name": deg_fp.name,
        }
        return sample


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch=14, out_ch=9, base_ch=64, dropout=0.0):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base_ch, dropout=dropout)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, dropout=dropout)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, dropout=dropout)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, dropout=dropout)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16, dropout=dropout)

        self.up4 = nn.Conv2d(base_ch * 16, base_ch * 8, 1)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8, dropout=dropout)

        self.up3 = nn.Conv2d(base_ch * 8, base_ch * 4, 1)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4, dropout=dropout)

        self.up2 = nn.Conv2d(base_ch * 4, base_ch * 2, 1)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, dropout=dropout)

        self.up1 = nn.Conv2d(base_ch * 2, base_ch, 1)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, dropout=dropout)

        self.out = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        u4 = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        u4 = self.up4(u4)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3(u3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(u2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(u1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))


        return torch.sigmoid(self.out(d1))


def masked_l1(pred, target, mask, channel_weights=None, eps=1e-6):


    diff = torch.abs(pred - target)

    if channel_weights is not None:
        w = torch.tensor(channel_weights, dtype=pred.dtype, device=pred.device)[None, :, None, None]
        diff = diff * w

    loss = (diff * mask).sum() / (mask.sum() * pred.shape[1] + eps)
    return loss


def gradient_loss(pred, target, mask, ch_idx=1, eps=1e-6):

    p = pred[:, ch_idx:ch_idx+1]
    t = target[:, ch_idx:ch_idx+1]

    p_dx = p[:, :, :, 1:] - p[:, :, :, :-1]
    t_dx = t[:, :, :, 1:] - t[:, :, :, :-1]
    m_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    p_dy = p[:, :, 1:, :] - p[:, :, :-1, :]
    t_dy = t[:, :, 1:, :] - t[:, :, :-1, :]
    m_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    loss_x = (torch.abs(p_dx - t_dx) * m_dx).sum() / (m_dx.sum() + eps)
    loss_y = (torch.abs(p_dy - t_dy) * m_dy).sum() / (m_dy.sum() + eps)

    return loss_x + loss_y


@torch.no_grad()
def evaluate(model, loader, device, channel_weights):
    model.eval()

    total = {
        "loss": 0.0,
        "l1_valid": 0.0,
        "l1_hard": 0.0,
        "height_mean_mae_valid": 0.0,
        "height_mean_mae_hard": 0.0,
        "density_mae_valid": 0.0,
        "density_mae_hard": 0.0,
        "boundary_mae_valid": 0.0,
        "boundary_mae_hard": 0.0,
        "n": 0,
    }

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        valid = batch["clean_valid"].to(device, non_blocking=True)
        hard = batch["clean_hard"].to(device, non_blocking=True)

        pred = model(x)

        l1_valid = masked_l1(pred, y, valid, channel_weights=channel_weights)
        l1_hard = masked_l1(pred, y, valid * hard, channel_weights=channel_weights)
        gl = gradient_loss(pred, y, valid, ch_idx=1)

        loss = l1_valid + 1.0 * l1_hard + 0.2 * gl

        def mae_ch(ch, mask):
            diff = torch.abs(pred[:, ch:ch+1] - y[:, ch:ch+1]) * mask
            return diff.sum() / (mask.sum() + 1e-6)

        bsz = x.shape[0]
        total["loss"] += float(loss.item()) * bsz
        total["l1_valid"] += float(l1_valid.item()) * bsz
        total["l1_hard"] += float(l1_hard.item()) * bsz



        total["height_mean_mae_valid"] += float(mae_ch(1, valid).item()) * bsz
        total["height_mean_mae_hard"] += float(mae_ch(1, valid * hard).item()) * bsz
        total["density_mae_valid"] += float(mae_ch(4, valid).item()) * bsz
        total["density_mae_hard"] += float(mae_ch(4, valid * hard).item()) * bsz
        total["boundary_mae_valid"] += float(mae_ch(8, valid).item()) * bsz
        total["boundary_mae_hard"] += float(mae_ch(8, valid * hard).item()) * bsz
        total["n"] += bsz

    n = max(total["n"], 1)
    return {k: v / n for k, v in total.items() if k != "n"}


def train(args):
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_set = ALS25DRestorationDataset(
        clean_dir=args.clean_dir,
        deg_dir=args.deg_dir,
        split="train",
        max_files=args.max_train_files,
        input_masks=True,
        degradation_tag=args.degradation_tag,
    )
    val_set = ALS25DRestorationDataset(
        clean_dir=args.clean_dir,
        deg_dir=args.deg_dir,
        split="val",
        max_files=args.max_val_files,
        input_masks=True,
        degradation_tag=args.degradation_tag,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = UNet2D(
        in_ch=14,
        out_ch=9,
        base_ch=args.base_ch,
        dropout=args.dropout,
    ).to(device)

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

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")



    channel_weights = [
        1.0,
        2.0,
        1.0,
        1.5,
        2.0,
        1.5,
        0.5,
        1.5,
        2.0,
    ]

    best_val = float("inf")
    best_epoch = None
    history = []

    print("[INFO] train samples:", len(train_set))
    print("[INFO] val samples:", len(val_set))
    print("[INFO] batch size:", args.batch_size)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        running = {
            "loss": 0.0,
            "l1_valid": 0.0,
            "l1_hard": 0.0,
            "grad": 0.0,
            "n": 0,
        }

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            valid = batch["clean_valid"].to(device, non_blocking=True)
            hard = batch["clean_hard"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(x)

                l1_valid = masked_l1(pred, y, valid, channel_weights=channel_weights)
                l1_hard = masked_l1(pred, y, valid * hard, channel_weights=channel_weights)
                gl = gradient_loss(pred, y, valid, ch_idx=1)

                loss = l1_valid + args.hard_weight * l1_hard + args.grad_weight * gl

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bsz = x.shape[0]
            running["loss"] += float(loss.item()) * bsz
            running["l1_valid"] += float(l1_valid.item()) * bsz
            running["l1_hard"] += float(l1_hard.item()) * bsz
            running["grad"] += float(gl.item()) * bsz
            running["n"] += bsz

            pbar.set_postfix({
                "loss": running["loss"] / running["n"],
                "l1": running["l1_valid"] / running["n"],
                "hard": running["l1_hard"] / running["n"],
            })

        scheduler.step()

        train_metrics = {
            k: v / max(running["n"], 1)
            for k, v in running.items()
            if k != "n"
        }

        val_metrics = evaluate(model, val_loader, device, channel_weights=channel_weights)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.time() - t0,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} "
            f"val_hard={row['val_l1_hard']:.6f} "
            f"val_height_hard={row['val_height_mean_mae_hard']:.6f} "
            f"val_density_hard={row['val_density_mae_hard']:.6f} "
            f"val_boundary_hard={row['val_boundary_mae_hard']:.6f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_metrics": val_metrics,
            "channel_weights": channel_weights,
        }

        torch.save(ckpt, out_dir / "last.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save(ckpt, out_dir / "best.pt")
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
            print(f"[BEST] epoch={epoch}, val_loss={best_val:.6f}")

    with open(
        out_dir / "training_complete.json",
        "w",
    ) as f:
        json.dump(
            {
                "status": "complete",
                "epochs_requested": args.epochs,
                "best_val_epoch": best_epoch,
                "best_val_loss": best_val,
                "degradation_tag": args.degradation_tag,
                "test_accessed": False,
            },
            f,
            indent=2,
        )

    print("[DONE]")
    print("[BEST VAL]", best_val)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--clean_dir", type=str, default="/root/autodl-tmp/ALS-DiffBridge/data/processed/dales_objects/maps_25d_b50_r050")
    parser.add_argument("--deg_dir", type=str, default="/root/autodl-tmp/ALS-DiffBridge/data/processed/dales_objects/maps_25d_b50_r050_degraded_v1")
    parser.add_argument("--out_dir", type=str, default="/root/autodl-tmp/ALS-DiffBridge/outputs/restoration_unet_b50_r050_v1")
    parser.add_argument(
        "--degradation_tag",
        type=str,
        default="deg_v1",
        help=(
            "Filename tag used by degraded 2.5D maps, e.g. "
            "deg_v3_medium."
        ),
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--base_ch", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hard_weight", type=float, default=1.0)
    parser.add_argument("--grad_weight", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
