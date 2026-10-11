"""Pure-PyTorch RandLA-Net backbone used when Open3D-ML is unavailable.

The official Open3D Windows wheels do not include the PyTorch extension.
RandLA-Net itself only needs standard PyTorch tensor operations once the
neighbour hierarchy has been prepared, as it is in this project.

The module layout and parameter names intentionally match Open3D-ML's
RandLANet implementation so checkpoints remain interchangeable.

Architecture adapted from the MIT-licensed Open3D-ML implementation:
https://github.com/isl-org/Open3D-ML/blob/main/ml3d/torch/models/randlanet.py
"""

from types import SimpleNamespace

import torch
import torch.nn as nn


class SharedMLP(nn.Module):
    """A 1x1 convolution followed by optional normalization and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        bn=True,
        activation_fn=None,
    ):
        super().__init__()
        conv_class = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = conv_class(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )
        self.batch_norm = (
            nn.BatchNorm2d(
                out_channels,
                eps=1e-6,
                momentum=0.01,
            )
            if bn
            else None
        )
        self.activation_fn = activation_fn

    def forward(self, input_tensor):
        output = self.conv(input_tensor)
        if self.batch_norm is not None:
            output = self.batch_norm(output)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output


class LocalSpatialEncoding(nn.Module):
    """Encode relative positions and neighbouring point features."""

    def __init__(
        self,
        dim_in,
        dim_out,
        num_neighbors,
        encode_pos=False,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(
            dim_in,
            dim_out,
            activation_fn=nn.LeakyReLU(0.2),
        )
        self.encode_pos = encode_pos

    @staticmethod
    def gather_neighbor(coords, neighbor_indices):
        batch_size, num_points, num_neighbors = (
            neighbor_indices.size()
        )
        dimension = coords.shape[2]
        extended_indices = neighbor_indices.unsqueeze(1).expand(
            batch_size,
            dimension,
            num_points,
            num_neighbors,
        )
        extended_coords = (
            coords.transpose(-2, -1)
            .unsqueeze(-1)
            .expand(
                batch_size,
                dimension,
                num_points,
                num_neighbors,
            )
        )
        return torch.gather(
            extended_coords,
            2,
            extended_indices,
        )

    def forward(
        self,
        coords,
        features,
        neighbor_indices,
        relative_features=None,
    ):
        batch_size, num_points, num_neighbors = (
            neighbor_indices.size()
        )

        if self.encode_pos:
            neighbor_coords = self.gather_neighbor(
                coords,
                neighbor_indices,
            )
            extended_coords = (
                coords.transpose(-2, -1)
                .unsqueeze(-1)
                .expand(
                    batch_size,
                    3,
                    num_points,
                    num_neighbors,
                )
            )
            relative_pos = extended_coords - neighbor_coords
            relative_dist = torch.sqrt(
                torch.sum(
                    torch.square(relative_pos),
                    dim=1,
                    keepdim=True,
                )
            )
            relative_features = torch.cat(
                [
                    relative_dist,
                    relative_pos,
                    extended_coords,
                    neighbor_coords,
                ],
                dim=1,
            )
        else:
            if relative_features is None:
                raise ValueError(
                    "relative_features are required on the "
                    "second spatial-encoding pass."
                )

        relative_features = self.mlp(relative_features)

        neighbor_features = self.gather_neighbor(
            features.transpose(1, 2).squeeze(3),
            neighbor_indices,
        )
        return (
            torch.cat(
                [neighbor_features, relative_features],
                dim=1,
            ),
            relative_features,
        )


class AttentivePooling(nn.Module):
    """Pool neighbour features with learned attention scores."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Softmax(dim=-2),
        )
        self.mlp = SharedMLP(
            in_channels,
            out_channels,
            activation_fn=nn.LeakyReLU(0.2),
        )

    def forward(self, input_tensor):
        scores = self.score_fn(
            input_tensor.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        features = torch.sum(
            scores * input_tensor,
            dim=-1,
            keepdim=True,
        )
        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):
    """RandLA-Net local feature aggregation block."""

    def __init__(self, dim_in, dim_out, num_neighbors):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.mlp1 = SharedMLP(
            dim_in,
            dim_out // 2,
            activation_fn=nn.LeakyReLU(0.2),
        )
        self.lse1 = LocalSpatialEncoding(
            10,
            dim_out // 2,
            num_neighbors,
            encode_pos=True,
        )
        self.pool1 = AttentivePooling(
            dim_out,
            dim_out // 2,
        )
        self.lse2 = LocalSpatialEncoding(
            dim_out // 2,
            dim_out // 2,
            num_neighbors,
        )
        self.pool2 = AttentivePooling(dim_out, dim_out)
        self.mlp2 = SharedMLP(dim_out, 2 * dim_out)
        self.shortcut = SharedMLP(dim_in, 2 * dim_out)
        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features, neighbor_indices):
        output = self.mlp1(features)
        output, neighbor_features = self.lse1(
            coords,
            output,
            neighbor_indices,
        )
        output = self.pool1(output)
        output, _ = self.lse2(
            coords,
            output,
            neighbor_indices,
            relative_features=neighbor_features,
        )
        output = self.pool2(output)
        return self.lrelu(
            self.mlp2(output) + self.shortcut(features)
        )


class RandLANet(nn.Module):
    """Open3D-ML-compatible RandLA-Net network definition."""

    def __init__(
        self,
        name="RandLANet",
        num_neighbors=16,
        num_layers=4,
        num_points=4096 * 11,
        num_classes=19,
        ignored_label_inds=None,
        sub_sampling_ratio=None,
        in_channels=3,
        dim_features=8,
        dim_output=None,
        grid_size=0.06,
        batcher="DefaultBatcher",
        ckpt_path=None,
        augment=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        del kwargs

        if ignored_label_inds is None:
            ignored_label_inds = [0]
        if sub_sampling_ratio is None:
            sub_sampling_ratio = [4, 4, 4, 4]
        if dim_output is None:
            dim_output = [16, 64, 128, 256]
        if augment is None:
            augment = {}

        self.cfg = SimpleNamespace(
            name=name,
            num_neighbors=num_neighbors,
            num_layers=num_layers,
            num_points=num_points,
            num_classes=num_classes,
            ignored_label_inds=ignored_label_inds,
            sub_sampling_ratio=sub_sampling_ratio,
            in_channels=in_channels,
            dim_features=dim_features,
            dim_output=dim_output,
            grid_size=grid_size,
            batcher=batcher,
            ckpt_path=ckpt_path,
            augment=augment,
        )
        self.device = torch.device(device)

        self.fc0 = nn.Linear(in_channels, dim_features)
        self.bn0 = nn.BatchNorm2d(
            dim_features,
            eps=1e-6,
            momentum=0.01,
        )

        encoder = []
        encoder_dim_list = []
        dim_feature = dim_features

        for layer_index in range(num_layers):
            encoder.append(
                LocalFeatureAggregation(
                    dim_feature,
                    dim_output[layer_index],
                    num_neighbors,
                )
            )
            dim_feature = 2 * dim_output[layer_index]
            if layer_index == 0:
                encoder_dim_list.append(dim_feature)
            encoder_dim_list.append(dim_feature)

        self.encoder = nn.ModuleList(encoder)
        self.mlp = SharedMLP(
            dim_feature,
            dim_feature,
            activation_fn=nn.LeakyReLU(0.2),
        )

        decoder = []
        for layer_index in range(num_layers):
            output_channels = encoder_dim_list[
                -layer_index - 2
            ]
            decoder.append(
                SharedMLP(
                    output_channels + dim_feature,
                    output_channels,
                    transpose=True,
                    activation_fn=nn.LeakyReLU(0.2),
                )
            )
            dim_feature = output_channels
        self.decoder = nn.ModuleList(decoder)

        self.fc1 = nn.Sequential(
            SharedMLP(
                dim_feature,
                64,
                activation_fn=nn.LeakyReLU(0.2),
            ),
            SharedMLP(
                64,
                32,
                activation_fn=nn.LeakyReLU(0.2),
            ),
            nn.Dropout(0.5),
            SharedMLP(
                32,
                num_classes,
                bn=False,
            ),
        )

    @staticmethod
    def random_sample(feature, pool_idx):
        feature = feature.squeeze(3)
        num_neighbors = pool_idx.size()[2]
        batch_size = feature.size()[0]
        dimension = feature.size()[1]
        pool_idx = torch.reshape(
            pool_idx,
            (batch_size, -1),
        )
        pool_idx = pool_idx.unsqueeze(2).expand(
            batch_size,
            -1,
            dimension,
        )
        feature = feature.transpose(1, 2)
        pool_features = torch.gather(
            feature,
            1,
            pool_idx,
        )
        pool_features = torch.reshape(
            pool_features,
            (
                batch_size,
                -1,
                num_neighbors,
                dimension,
            ),
        )
        pool_features, _ = torch.max(
            pool_features,
            2,
            keepdim=True,
        )
        return pool_features.permute(0, 3, 1, 2)

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        feature = feature.squeeze(3)
        dimension = feature.size(1)
        batch_size = interp_idx.size()[0]
        up_num_points = interp_idx.size()[1]
        interp_idx = torch.reshape(
            interp_idx,
            (batch_size, up_num_points),
        )
        interp_idx = interp_idx.unsqueeze(1).expand(
            batch_size,
            dimension,
            -1,
        )
        interpolated = torch.gather(
            feature,
            2,
            interp_idx,
        )
        return interpolated.unsqueeze(3)

    def forward(self, inputs):
        features = inputs["features"].to(self.device)
        coords_list = [
            array.to(self.device)
            for array in inputs["coords"]
        ]
        neighbor_indices_list = [
            array.to(self.device)
            for array in inputs["neighbor_indices"]
        ]
        subsample_indices_list = [
            array.to(self.device)
            for array in inputs["sub_idx"]
        ]
        interpolation_indices_list = [
            array.to(self.device)
            for array in inputs["interp_idx"]
        ]

        features = (
            self.fc0(features)
            .transpose(-2, -1)
            .unsqueeze(-1)
        )
        features = self.bn0(features)
        features = nn.functional.leaky_relu(
            features,
            negative_slope=0.2,
        )

        encoder_features = []
        for layer_index in range(self.cfg.num_layers):
            encoded = self.encoder[layer_index](
                coords_list[layer_index],
                features,
                neighbor_indices_list[layer_index],
            )
            sampled = self.random_sample(
                encoded,
                subsample_indices_list[layer_index],
            )
            if layer_index == 0:
                encoder_features.append(encoded.clone())
            encoder_features.append(sampled.clone())
            features = sampled

        features = self.mlp(features)
        for layer_index in range(self.cfg.num_layers):
            interpolated = self.nearest_interpolation(
                features,
                interpolation_indices_list[
                    -layer_index - 1
                ],
            )
            decoded = torch.cat(
                [
                    encoder_features[-layer_index - 2],
                    interpolated,
                ],
                dim=1,
            )
            features = self.decoder[layer_index](decoded)

        scores = self.fc1(features)
        return scores.squeeze(3).transpose(1, 2)
