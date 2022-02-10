from units.net import CondUDA
from data.dataloader import GeoDataLoader
from torch.utils import data
import torch.nn as nn
import torch
import argparse


LEARNING_RATE_G = 0.001
LEARNING_RATE_D = 0.0001
TOTAL_STEPS = 100000
DECAY_STEP = int(TOTAL_STEPS * .75)


def argument_parser():
    pass


args = argument_parser()


def ce_loss():
    pass


def lr_decay(base_lr, iter, max_iter, decay_step):
    return base_lr * ((max_iter - float(iter)) / float((max_iter - decay_step)))


def adjust_lr_G(optimizer, i_iter):
    lr = lr_decay(args.learning_rate_g, i_iter, args.total_steps, args.decay_step)
    optimizer.param_groups[0]['lr'] = lr


def adjust_lr_D(optimizer, i_iter):
    lr = lr_decay(args.learning_rate_d, i_iter, args.total_steps, args.decay_step)
    optimizer.param_groups[0]['lr'] = lr


def train():
    pass
# args = argument_parser()
#
# segm_loss = nn.CrossEntropyLoss()
# adv_loss = nn.BCEWithLogitsLoss()
#
# interp = torch.nn.Upsample(size=(int(args.input_size_source[0]), int(args.input_size_source[1])),
#                            mode='bilinear',
#                            align_corners=False)
#
# learning_rate_g = args.learning_rate_G
# learning_rate_d = args.learning_rate_D
# total_steps = args.max_iter
# decay_step = int(total_steps * 0.75)
#
# # Define optimizers:
# params = list(self.net_A_enc.parameters()) + list(self.net_A_dec.parameters()) + \
#          list(self.net_B_enc.parameters()) + list(self.net_B_dec.parameters())
#
# self.optimizer_g = torch.optim.Adam(params, lr=self.learning_rate_g, betas=(0.5, 0.999))
# self.optimizer_d = torch.optim.Adam(itertools.chain(self.net_d_A.parameters(), self.net_d_B.parameters()),
#                                     lr=self.learning_rate_d, betas=(0.5, 0.999))
#
#
# source_data_loader = data.DataLoader(GeoDataLoader(x_data=args.source,
#                                                    size=(int(args.input_size_source[0]),
#                                                          int(args.input_size_source[1]))),
#                                      batch_size=args.batch_size,
#                                      shuffle=True,
#                                      num_workers=0,
#                                      pin_memory=True,
#                                      drop_last=True)
#
# target_data_loader = data.DataLoader(GeoDataLoader(x_data=args.target,
#                                                    size=(int(args.input_size_target[0]),
#                                                          int(args.input_size_target[1]))),
#                                      batch_size=args.batch_size,
#                                      shuffle=True,
#                                      num_workers=0,
#                                      pin_memory=True,
#                                      drop_last=True)



