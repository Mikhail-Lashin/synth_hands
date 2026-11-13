from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class HandPoseIPR(nn.Module):
    """
    Integral Pose Regression (IPR) network for 3D hand keypoints estimation
    from a single-channel depth map.
    """

    def __init__(self, num_joints: int = 21):
        super().__init__()
        self.num_joints = num_joints
        backbone = models.resnet18(weights=None)

        # 1st conv - 1 channel depth
        old_conv = backbone.conv1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # deconv head for heatmaps and depthmaps
        self.deconv_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_joints * 2, kernel_size=1),
        )

        self.coord_scale = nn.Parameter(torch.tensor([0.18, 0.18, 0.12]))
        self.coord_bias = nn.Parameter(torch.tensor([0.00, 0.00, 0.22]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 1, H, W)
        x_feat = self.conv1(x)
        x_feat = self.bn1(x_feat)
        x_feat = self.relu(x_feat)
        x_feat = self.maxpool(x_feat)

        x_feat = self.layer1(x_feat)
        x_feat = self.layer2(x_feat)
        x_feat = self.layer3(x_feat)
        x_feat = self.layer4(x_feat)

        out = self.deconv_head(x_feat)

        heatmaps = out[:, : self.num_joints]
        depthmaps = out[:, self.num_joints :]

        B, K, H, W = heatmaps.shape
        prob = F.softmax(heatmaps.view(B, K, -1), dim=-1).view(B, K, H, W)

        grid_x = torch.linspace(-1.0, 1.0, W, device=x.device).view(1, 1, 1, W)
        grid_y = torch.linspace(-1.0, 1.0, H, device=x.device).view(1, 1, H, 1)

        pred_x = (prob * grid_x).sum(dim=(-2, -1))
        pred_y = (prob * grid_y).sum(dim=(-2, -1))
        pred_z = (prob * torch.tanh(depthmaps)).sum(dim=(-2, -1))

        norm_joints = torch.stack([pred_x, pred_y, pred_z], dim=-1)
        joints_3d = norm_joints * self.coord_scale + self.coord_bias

        return joints_3d


class HandPoseResNet18(nn.Module):
    """
    Baseline direct regression network for 3D hand keypoints
    using ResNet18 backbone.
    """

    def __init__(self, num_joints: int = 21):
        super().__init__()
        self.num_joints = num_joints
        self.backbone = models.resnet18(weights=None)
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_joints * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return out.view(-1, self.num_joints, 3)
