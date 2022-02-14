from units.backbone import Encoder, Decoder
from units.generator import CondGenerator
from data.dataloader import GeoDataLoader
from units.discriminator import FCDiscriminator
import torch.nn.functional as F
from torch.utils import data
import torch.nn as nn
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
torch.autograd.set_detect_anomaly(True)


class CondUDAModel(nn.Module):
    def __init__(self, args):
        super(CondUDAModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Define the model's modules:
        self.encoder = Encoder()
        self.decoder = Decoder(5)
        self.generator = CondGenerator(device=self.device, scale=32)
        if args.model_mode == 'train':
            self.net_d = FCDiscriminator(512*(args.input_size_source[0] // 32)**2)

        # Define the losses:
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.interp = torch.nn.Upsample(size=(int(args.input_size_source[0]), int(args.input_size_source[1])),
                                        mode='bilinear',
                                        align_corners=False)

        self.learning_rate_g = args.learning_rate_G
        self.learning_rate_d = args.learning_rate_D
        self.total_steps = args.max_iter
        self.decay_step = int(self.total_steps * 0.75)

        self.source_domain = 0
        self.target_domain = 1

        self.ignore_label = 255

        # Define optimizers:
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
                 list(self.generator.parameters())

        self.optimizer_g = torch.optim.Adam(params, lr=self.learning_rate_g, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.net_d.parameters(), lr=self.learning_rate_d, betas=(0.5, 0.999))

        if args.model_mode == 'train':
            self.source_data_loader = data.DataLoader(GeoDataLoader(x_data=args.source,
                                                                    y_data=args.source_labels,
                                                                    resize=(args.input_size_source[0], args.input_size_source[1])),
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      drop_last=True)

            self.target_data_loader = data.DataLoader(GeoDataLoader(x_data=args.target,
                                                                    y_data=args.target_labels,
                                                                    resize=(args.input_size_target[0], args.input_size_target[1])),
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=0,
                                                      pin_memory=True,
                                                      drop_last=True)

        else:
            self.val_data_loader = data.DataLoader(GeoDataLoader(x_data=args.target,
                                                                 y_data=args.target_labels,
                                                                 resize=(int(args.input_size_source[0]), int(args.input_size_source[1]))),
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

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
        return hist

    def batch_metrics(self, predictions, gts):
        batch_hist = np.zeros((self.n_classes, self.n_classes))
        label_pred = predictions.clone().cpu().detach().numpy().copy()
        label_true = gts.clone().cpu().detach().numpy().copy()
        label_pred = np.asarray(np.argmax(label_pred, axis=1), dtype=np.uint8).copy()

        for lp, lt in zip(label_pred, label_true):
            batch_hist += self._fast_hist(lp.flatten(), lt.flatten())
        acc = np.diag(batch_hist).sum() / batch_hist.sum()
        iu = np.diag(batch_hist) / (batch_hist.sum(axis=1) + batch_hist.sum(axis=0) - np.diag(batch_hist))
        mean_iu = np.nanmean(iu)
        return mean_iu, acc, batch_hist

    def per_class_iu(self, hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def segm_loss_cals(self, pred, label, weight=None):
        label = label.clone().detach().type(dtype=torch.long).cuda(device=self.device)
        n, c, h, w = pred.size()

        target_mask = (label >= 0) * (label != self.ignore_label)
        target = label[target_mask]
        predict = pred.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)

        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        return loss

    def set_inputs(self, x, x_l, y, y_l):
        self.real_A = x.float().to(self.device)
        self.real_A_labels = x_l.to(self.device)
        self.real_B = y.float().to(self.device)
        self.real_B_labels = y_l.to(self.device)

    def forward(self):
        self.feat1, self.feat2, self.feat3, self.feat4 = self.encoder(self.real_A)
        gen_out = self.generator(self.feat1)
        self.feat4 = gen_out + self.feat4
        self.out = self.decoder([self.feat1, self.feat2, self.feat3, self.feat4])
        self.out = self.interp(self.out)
        _, _, _, self.feat4_B = self.encoder(self.real_B)


    def backward_G(self):
        self.G_loss_segm = self.segm_loss_cals(self.out, self.real_A_labels)
        d_out_A = self.net_d(self.feat4)

        self.G_loss_adv = self.adv_loss(d_out_A,
                                        torch.FloatTensor(d_out_A.data.size()).fill_(self.target_domain).cuda())
        self.gen_loss = self.G_loss_segm + self.G_loss_adv
        self.gen_loss.backward()

    def backward_D(self):
        feat4 = self.feat4.detach()
        feat4_B = self.feat4_B.detach()
        d_out_A = self.net_d(feat4)
        d_out_B = self.net_d(feat4_B)

        D_loss_adv_A = self.adv_loss(d_out_A,
                                        torch.FloatTensor(d_out_A.data.size()).fill_(self.source_domain).cuda())
        D_loss_adv_B = self.adv_loss(d_out_B,
                                        torch.FloatTensor(d_out_B.data.size()).fill_(self.target_domain).cuda())
        self.D_loss_adv = D_loss_adv_A + D_loss_adv_B
        self.D_loss_adv.backward()

    def optimize_parameters(self):
        # Forward pass:
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        # Step 1: Update Decoder and Discriminator (fixed Encoder and Generator):
        # self.set_requires_grad([self.encoder, self.generator], False)
        # self.set_requires_grad([self.decoder, self.net_d], True)
        self.forward()
        self.backward_G()
        self.backward_D()
        self.optimizer_g.step()
        self.optimizer_d.step()

        # Step 2: Update Encoder and Generator (fixed Decoder and Discriminator):
        self.set_requires_grad([self.encoder, self.generator], True)
        self.set_requires_grad([self.decoder, self.net_d], False)
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        self.forward()
        self.backward_G()
        self.backward_D()
        self.optimizer_g.step()
        self.optimizer_d.step()

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

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.generator.to(self.device)
        self.net_d.to(self.device)

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
            source_images, source_labels, _ = source_batch
            target_images, target_labels, _ = target_batch

            self.set_inputs(source_images, source_labels, target_images, target_labels)
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
    parser.add_argument("--source_labels", type=str, default=params['y_source'],
                        help="Define a path to the source labels.")
    parser.add_argument("--target_labels", type=str, default=params['y_target'],
                        help="Define a path to the source labels.")
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
        's_size': (256, 256),
        't_size': (256, 256),
        'batch_size': 1,
        'x_source': r"D:\PhyCode\data\geo\geodata\new_experiments\BC5_full",
        'y_source': r'D:\PhyCode\data\geo\geodata\new_experiments\source_labels',
        'x_target': r"D:\PhyCode\data\geo\geodata\new_experiments\extra_target\M1\M1_full",
        'y_target': r'D:\PhyCode\data\geo\geodata\new_experiments\extra_target\M1_labels',
        'mode': 'train',
        'experiment_name': 'BC5_to_M1',
        'iters': 100000
    }

    args = get_arguments(params)

    model = CondUDAModel(args)

    if args.model_mode == 'train':
        model.train_model()
    else:
        model.test_and_save()
