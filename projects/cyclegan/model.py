"""CycleGAN generator architecture used by the trained notebook models."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int, padding_mode: str = "reflect") -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.InstanceNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.InstanceNorm2d(num_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class Generator(nn.Module):
    """Nine-block ResNet generator from the 256px CycleGAN setup."""

    def __init__(self, padding_mode: str = "reflect") -> None:
        super().__init__()

        blocks: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(
                    3,
                    64,
                    kernel_size=7,
                    padding=3,
                    padding_mode=padding_mode,
                ),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),
            )
        ]

        for index in range(2):
            in_channels = 64 * (2**index)
            out_channels = in_channels * 2
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        padding_mode=padding_mode,
                    ),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        out_channels = 256
        blocks.extend(ResidualBlock(out_channels, padding_mode) for _ in range(9))

        for _ in range(2):
            in_channels = out_channels
            out_channels = in_channels // 2
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        blocks.append(
            nn.Sequential(
                nn.Conv2d(
                    out_channels,
                    3,
                    kernel_size=7,
                    padding=3,
                    padding_mode=padding_mode,
                ),
                nn.Tanh(),
            )
        )

        self.model = nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)
