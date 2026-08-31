"""GeneratorA from the DisGUIDE reference implementation (disguide-main)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def img_to_grayscale(img: torch.Tensor, freq: int) -> torch.Tensor:
    """Turn images to grayscale with frequency 1 / param."""
    if freq == 0:
        return img

    img_clone = img.clone()
    img_d = img_clone.detach()
    rng_chan = torch.randint(3, (1,))[0]
    img_clone[:, (rng_chan + 1) % 3] = img_d[:, rng_chan]
    img_clone[:, (rng_chan + 2) % 3] = img_d[:, rng_chan]
    return img_clone


class GeneratorA(nn.Module):
    """Reference DisGUIDE generator (network/gan.py::GeneratorA)."""

    def __init__(
        self,
        nz: int = 100,
        ngf: int = 64,
        nc: int = 1,
        img_size: int = 32,
        activation: torch.nn.Module | None = None,
        final_bn: bool = True,
        grayscale: int = 0,
    ) -> None:
        super().__init__()
        self.grayscale = grayscale
        if activation is None:
            raise ValueError("Provide a valid activation function")
        self.activation = activation

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if final_bn:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                nn.BatchNorm2d(nc, affine=False),
            )
        else:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            )

    def forward(self, z: torch.Tensor, pre_x: bool = False) -> torch.Tensor:
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)

        if pre_x:
            if self.grayscale != 0:
                raise NotImplementedError()
            return img
        if self.grayscale != 0:
            return img_to_grayscale(self.activation(img), self.grayscale)
        return self.activation(img)
