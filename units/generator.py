import torch
import torch.nn as nn
import numpy as np


class GenBlocks(nn.Module):
    def __init__(self, n_filters=64, n_blocks=16):
        super(GenBlocks, self).__init__()
        self.blocks = []
        for _ in range(n_blocks):
            block = nn.Sequential(*[
                nn.Conv2d(n_filters, n_filters, kernel_size=(1, 1)),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.Conv2d(n_filters, n_filters, kernel_size=(1, 1))
            ])
            self.blocks.append(block)

    def forward(self, x):
        for module in self.blocks:
            x += module(x)
        return x


class CondGenerator(nn.Module):
    def __init__(self):
        super(CondGenerator, self).__init__()
        self.gen_blocks = GenBlocks(n_filters=64, n_blocks=16)
        self.conv = nn.Conv2d(65, 64, kernel_size=(1, 1))
        self.conv_out = nn.Conv2d(64, 512, 2, 2)

    def forward(self, x):
        noise = torch.from_numpy(np.random.uniform(low=-1, high=1, size=(x.shape[0], 1, x.shape[2], x.shape[3])))
        noise = noise.type(torch.float)
        x = torch.cat([noise, x], 1)
        x = self.conv(x)
        x = self.gen_blocks(x)
        x = self.conv_out(x)
        x = nn.AvgPool2d(8)(x)
        return x


if __name__ == "__main__":
    image = np.random.random(size=(1, 64, 512, 512))
    image = torch.Tensor(image)
    gen = CondGenerator()
    out = gen(image)
    # print(out.shape)
