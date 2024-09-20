import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict

class ResBlock(nn.Module):
    def __init__(self, ncf, use_bias=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ncf, ncf, kernel_size=3, stride=1, padding=1, bias=use_bias)),
            ('bn', nn.InstanceNorm2d(ncf)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ncf, ncf, kernel_size=3, stride=1, padding=1, bias=use_bias)),
            ('bn', nn.InstanceNorm2d(ncf)),
        ]))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        out = self.relu(out)
        return out

class Res_Generator(pl.LightningModule):
    def __init__(self, ngf, num_resblock, lr=1e-4):
        super(Res_Generator, self).__init__()
        self.lr = lr
        self.save_hyperparameters()

        self.conv1 = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(6, ngf, kernel_size=7, stride=1, padding=0, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf * 2)),
            ('relu', nn.ReLU(inplace=True)),
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf * 4)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.num_resblock = num_resblock
        for i in range(num_resblock):
            setattr(self, 'res' + str(i + 1), ResBlock(ngf * 4, use_bias=True))

        self.deconv3 = nn.Sequential(OrderedDict([
            ('deconv', nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf * 2)),
            ('relu', nn.ReLU(True)),
        ]))
        self.deconv2 = nn.Sequential(OrderedDict([
            ('deconv', nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ngf)),
            ('relu', nn.ReLU(True)),
        ]))
        self.deconv1 = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(ngf, 3, kernel_size=7, stride=1, padding=0, bias=False)),
            ('tanh', nn.Tanh()),
        ]))

    def forward(self, im, pose):
        x = torch.cat((im, pose), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        for i in range(self.num_resblock):
            res = getattr(self, 'res' + str(i + 1))
            x = res(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class DC_Discriminator(pl.LightningModule):
    def __init__(self, ndf, num_att, lr=1e-4):
        super(DC_Discriminator, self).__init__()
        self.lr = lr
        self.save_hyperparameters()

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 2)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 4)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 8)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv5 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 8)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv6 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 8)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.dis = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=0, bias=False)),
        ]))
        self.att = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(7 * 3 * ndf * 8, 1024)),
            ('relu', nn.ReLU(True)),
            ('fc2', nn.Linear(1024, num_att))
        ]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        dis = self.dis(x)
        x = x.view(x.size(0), -1)
        att = self.att(x)
        return dis, att

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class Patch_Discriminator(pl.LightningModule):
    def __init__(self, ndf, lr=1e-4):
        super(Patch_Discriminator, self).__init__()
        self.lr = lr
        self.save_hyperparameters()

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 2)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 4)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=0, bias=True)),
            ('bn', nn.InstanceNorm2d(ndf * 8)),
            ('relu', nn.LeakyReLU(0.2, True))
        ]))
        self.dis = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)),
        ]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        dis = self.dis(x).squeeze()
        return dis

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class GANModel(pl.LightningModule):
    def __init__(self, generator, discriminator):
        super(GANModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.criterion_gan = nn.MSELoss()  # Example loss

    def forward(self, im, pose):
        return self.generator(im, pose)

    def training_step(self, batch, batch_idx, optimizer_idx):
        src_img, pose, tgt_img = batch

        # Generator update
        if optimizer_idx == 0:
            fake_img = self(src_img, pose)
            dis_fake, _ = self.discriminator(fake_img)
            g_loss = self.criterion_gan(dis_fake, torch.ones_like(dis_fake))
            self.log('g_loss', g_loss)
            return g_loss

        # Discriminator update
        if optimizer_idx == 1:
            fake_img = self(src_img, pose)
            dis_fake, _ = self.discriminator(fake_img.detach())
            dis_real, _ = self.discriminator(tgt_img)
            d_loss_fake = self.criterion_gan(dis_fake, torch.zeros_like(dis_fake))
            d_loss_real = self.criterion_gan(dis_real, torch.ones_like(dis_real))
            d_loss = (d_loss_fake + d_loss_real) / 2
            self.log('d_loss', d_loss)
            return d_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        return [opt_g, opt_d], []

