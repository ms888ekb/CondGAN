from units.net import CondUDA
from data.dataloader import GeoDataLoader
from torch.utils import data
import torch.nn as nn
import torch
import argparse


def argument_parser():
    pass

# args = argument_parser()


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


if __name__ == "__main__":
    pass