from units.backbone import Encoder, Decoder
from units.generator import CondGenerator
from data.dataloader import GeoDataLoader
from units.discriminator import FCDicriminator
from torch.utils import data
import torch.nn as nn
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np


class CondUDAModel(nn.Module):
    def __init__(self, args):
        super(CondUDAModel, self).__init__()
        # Define the model's modules:
        self.encoder = Encoder()
        self.decoder = Decoder(5)
        self.generator = CondGenerator()
        if args.model_mode == 'train':
            self.net_d = FCDicriminator(32*32*512)

        # Define the losses:
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.interp = torch.nn.Upsample(size=(int(args.input_size_source[0]), int(args.input_size_source[1])),
                                        mode='bilinear',
                                        align_corners=False)

        self.learning_rate_g = args.learning_rate_G
        self.learning_rate_d = args.learning_rate_D
        self.total_steps = args.max_iter
        self.decay_step = int(self.total_steps * 0.75)

        # Define optimizers:
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
                 list(self.generator.parameters())

        self.optimizer_g = torch.optim.Adam(params, lr=self.learning_rate_g, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=self.learning_rate_d, betas=(0.5, 0.999))

        if args.model_mode == 'train':
            self.source_data_loader = data.DataLoader(GeoDataLoader(x_data=args.source,
                                                                    size=(int(args.input_size_source[0]), int(args.input_size_source[1]))),
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      drop_last=True)

            self.target_data_loader = data.DataLoader(GeoDataLoader(x_data=args.target,
                                                                    size=(int(args.input_size_target[0]), int(args.input_size_target[1]))),
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      drop_last=True)

        else:
            self.val_data_loader = data.DataLoader(GeoDataLoader(x_data=args.source,
                                                                 size=(int(args.input_size_source[0]), int(args.input_size_source[1]))),
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   pin_memory=True,
                                                   drop_last=True)

        self.writer = SummaryWriter(log_dir=os.path.join('runs/SRSemI2I/', args.name))
        self.ename = args.name
        os.makedirs(os.path.join('CondUDA/', self.ename), exist_ok=True)

    def lr_decay(self, base_lr, iter, max_iter, decay_step):
        return base_lr * ((max_iter - float(iter)) / float((max_iter - decay_step)))

    def adjust_learning_rate_G(self, optimizer, i_iter):
        lr = self.lr_decay(self.learning_rate_g, i_iter, self.total_steps, self.decay_step)
        optimizer.param_groups[0]['lr'] = lr

    def adjust_learning_rate_D(self, optimizer, i_iter):
        lr = self.lr_decay(self.learning_rate_d, i_iter, self.total_steps, self.decay_step)
        optimizer.param_groups[0]['lr'] = lr

    def set_inputs(self, x, y):
        # Get x and y batch on form: [b x c x h x w]
        self.real_A = x.cuda()
        self.real_B = y.cuda()

    def forward(self):
        pass

    def backward_G(self):
        pass

    def backward_D(self):
        pass

    def optimize_parameters(self):
        # Forward pass:
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        pass

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_model(self):
        self.encoder.train()
        self.decoder.train()
        self.generator.train()
        self.net_d.train()

        self.encoder.cuda()
        self.decoder.cuda()
        self.generator.cuda()
        self.net_d.cuda()

        target_iterator = enumerate(self.target_data_loader)
        source_iterator = enumerate(self.source_data_loader)

        for step in range(self.total_steps):
            if step >= self.decay_step:
                self.adjust_learning_rate_G(self.optimizer_g, step)
                self.adjust_learning_rate_D(self.optimizer_d, step)

            # Get source batch:
            try:
                _, source_batch = next(source_iterator)
            except StopIteration:
                source_iterator = enumerate(self.source_data_loader)
                _, source_batch = next(source_iterator)

            # Get target batch:
            try:
                _, target_batch = next(target_iterator)
            except StopIteration:
                target_iterator = enumerate(self.target_data_loader)
                _, target_batch = next(target_iterator)

            # Unpack batches:
            source_images, s_geo_ref = source_batch
            target_images, t_geo_ref = target_batch

            self.set_inputs(source_images, target_images)
            self.optimize_parameters()

            if step % 5 == 0:
                # self.writer.add_image('self_rec_A', srecA)
                # self.informer()
                self.writer.add_scalar('adv_loss', self.adv_loss, step)
                self.writer.add_scalar('segm_loss', self.segm_loss, step)

            if step == self.total_steps-1:
                print(f'Saving the model at step {step+1}...')
                torch.save(self.net_A_enc.state_dict(),
                           os.path.join(*['CondUDA/', self.ename + '/CondUDA_encoder.pth']))
                torch.save(self.net_B_enc.state_dict(),
                           os.path.join(*['CondUDA/', self.ename + '/CondUDA_decoder.pth']))
                torch.save(self.net_A_dec.state_dict(),
                           os.path.join(*['CondUDA/', self.ename + '/CondUDA_generator.pth']))
                torch.save(self.net_B_dec.state_dict(),
                           os.path.join(*['CondUDA/', self.ename + '/CondUDA_disc.pth']))

    def test_and_save(self):
        pass

    def back_transformation(self, image_tensor, all=False):
        img = image_tensor.detach()
        image_numpy = img[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        if all:
            return image_numpy.astype(np.uint8)
        else:
            image_numpy = image_numpy[:, :, :3]
            image_numpy = image_numpy[:, :, ::-1]
            image_numpy = image_numpy.transpose(2, 0, 1)
            return image_numpy.astype(np.uint8)


def get_arguments(params):
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SemI2I Network")
    parser.add_argument("--source", type=str, default=params['x_source'],
                        help="Define a path to the source images.")
    parser.add_argument("--target", type=str, default=params['x_target'],
                        help="Define a path to the source images.")
    parser.add_argument("--batch-size", type=int, default=params['batch_size'],
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--max_iter", type=int, default=params['iters'],
                        help="Number of training steps.")
    parser.add_argument("--input-size-source", type=str, default=params['s_size'],
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--input-size-target", type=str, default=params['t_size'],
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--learning-rate-G", type=float, default=0.0001,
                        help="Base learning rate for generator.")
    parser.add_argument("--learning-rate-D", type=float, default=0.00002,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--model-mode", type=str, default=params['mode'],
                        help="Choose the model mode: train/test")
    parser.add_argument("--name", type=str, default=params['experiment_name'],
                        help="Define the name of the experiment.")

    return parser.parse_args()


if __name__ == "__main__":
    params = {
        's_size': (512, 512),
        't_size': (512, 512),
        'batch_size': 1,
        'x_source': r"/data/0_source_full",
        'x_target': r"/data/0_target_full",
        'mode': 'train',
        'experiment_name': 'GE1_to_WV2',
        'iters': 25000
    }

    args = get_arguments(params)

    model = CondUDAModel(args)

    if args.model_mode == 'train':
        model.train_model()
    else:
        model.test_and_save()
