import torch.nn as nn
from units.backbone import Encoder, Decoder
from units.generator import CondGenerator


class CondUDA(nn.Module):
    def __init__(self):
        super(CondUDA, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(5)
        self.generator = CondGenerator()

    def forward(self, x, use_gen=True):
        if use_gen:
            features = self.encoder(x)
            gen_out = self.generator(features[0])
            features[1] = gen_out + features[1]
            out = self.decoder(features)
        else:
            features = self.encoder(x)
            out = self.decoder(x)
        return out, features[1]


if __name__ == '__main__':
    pass
