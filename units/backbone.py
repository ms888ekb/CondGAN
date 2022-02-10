import torch
import torchvision
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        self.backbone = list(vgg19.children())[:36][0]
        self.get_features = [0, 18, 27, 36]

    def forward(self, x):
        features = []
        for i, submodule in enumerate(self.backbone.children()):
            x = submodule(x)
            if i in self.get_features:
                features.append(x)
        return features


class Decoder(nn.Module):
    def __init__(self, n_classes):
        super(Decoder, self).__init__()
        self.score_last = nn.Conv2d(512, n_classes, kernel_size=(1, 1))
        self.score_pool4 = nn.Conv2d(512, n_classes, kernel_size=(1, 1))
        self.score_pool3 = nn.Conv2d(256, n_classes, kernel_size=(1, 1))

        self.upconv1 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.upconv2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.upconv3 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=(16, 16), stride=(8, 8), bias=False)

    def forward(self, features):
        score1 = self.score_last(features[-1])
        x = self.upconv1(score1)

        score2 = self.score_pool4(features[-2])
        x = torch.nn.Upsample(size=(score2.shape[2], score2.shape[3]))(x)
        x = x + score2
        x = self.upconv2(x)

        score3 = self.score_pool3(features[-3])
        x = torch.nn.Upsample(size=(score3.shape[2], score3.shape[3]))(x)
        x = x + score3
        x = self.upconv3(x)

        return x


if __name__ == "__main__":
    image = np.random.random(size=(1, 3, 512, 512))
    image = torch.Tensor(image)
    enc = Encoder()
    dec = Decoder(5)
    feat = enc(image)
    out = dec(feat)
    print(out.shape)


