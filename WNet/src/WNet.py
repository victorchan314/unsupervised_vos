import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableDownsampleConv(nn.Module):
    def __init__(self, n, kernel_size=3):
        super().__init__()
        self.n = n
        self.kernel_size = kernel_size
        self.initialize_layers()

    def initialize_layers(self):
        layers = [nn.Conv2d(self.n, 2 * self.n, 1),
                nn.Conv2d(2 * self.n, 2 * self.n, self.kernel_size, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(2 * self.n),
                nn.Conv2d(2 * self.n, 2 * self.n, 1),
                nn.Conv2d(2 * self.n, 2 * self.n, self.kernel_size, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(2 * self.n)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SeparableUpsampleConv(nn.Module):
    def __init__(self, n, kernel_size=3):
        super().__init__()
        self.n = n
        self.kernel_size = kernel_size
        self.initialize_layers()

    def initialize_layers(self):
        layers = [nn.Conv2d(2 * self.n, self.n, 1),
                nn.Conv2d(self.n, self.n, self.kernel_size, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(self.n),
                nn.Conv2d(self.n, self.n, 1),
                nn.Conv2d(self.n, self.n, self.kernel_size, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(self.n)]
        self.layers = nn.Sequential(*layers)

    def forward(self, skip, x):
        return self.layers(torch.cat((skip, x), dim=1))

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.initialize_layers()

    def initialize_layers(self):
        layers = [nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(self.out_channels),
                nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(self.out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class UEncoder(nn.Module):
    def __init__(self, n_modules=9, kernel_size=3, k=3):
        super().__init__()
        if not n_modules % 2 == 1:
            raise ValueError("n_modules was {}, but must be odd".format(n_modules))
        self.n_modules = n_modules
        self.kernel_size = kernel_size
        self.k = k
        self.initialize_layers()

    def initialize_layers(self):
        self.contract = nn.ModuleList()
        self.contract.append(ConvModule(3, 64))
        for i in range(self.n_modules // 2):
            self.contract.append(SeparableDownsampleConv(64 * (2**i), kernel_size=self.kernel_size))

        self.expand = nn.ModuleList()
        for i in range(self.n_modules // 2 - 1):
            self.expand.append(SeparableUpsampleConv(512 // (2**i), kernel_size=self.kernel_size))
        self.deconvs = nn.ModuleList()
        for i in range(self.n_modules // 2):
            self.deconvs.append(nn.ConvTranspose2d(512 // (2**i) * 2, 512 // (2**i), self.kernel_size + 1, stride=2, padding=1))

        self.final_conv_module = ConvModule(128, 64)
        self.final_conv = nn.Conv2d(64, self.k, kernel_size=1)

    def forward(self, x):
        out = x
        intermediate_values = []
        for i, layer in enumerate(self.contract):
            out = layer(out)
            intermediate_values.append(out)
            if i != len(self.contract) - 1:
                out = F.max_pool2d(out, 2)
        for i, layer in enumerate(self.expand):
            out = self.deconvs[i](out)
            out = layer(intermediate_values[-(i + 2)], out)
        out = self.deconvs[-1](out)

        out = self.final_conv_module(torch.cat((intermediate_values[0], out), dim=1))
        out = self.final_conv(out)
        out = F.softmax(out, dim=1)

        return out

class UDecoder(nn.Module):
    def __init__(self, n_modules=9, kernel_size=3, k=3):
        super().__init__()
        if not n_modules % 2 == 1:
            raise ValueError("n_modules was {}, but must be odd".format(n_modules))
        self.n_modules = n_modules
        self.kernel_size = kernel_size
        self.k = k
        self.initialize_layers()

    def initialize_layers(self):
        self.contract = nn.ModuleList()
        self.contract.append(ConvModule(self.k, 64))
        for i in range(self.n_modules // 2):
            self.contract.append(SeparableDownsampleConv(64 * (2**i), kernel_size=self.kernel_size))

        self.expand = nn.ModuleList()
        for i in range(self.n_modules // 2 - 1):
            self.expand.append(SeparableUpsampleConv(512 // (2**i), kernel_size=self.kernel_size))

        self.deconvs = nn.ModuleList()
        for i in range(self.n_modules // 2):
            self.deconvs.append(nn.ConvTranspose2d(512 // (2**i) * 2, 512 // (2**i), self.kernel_size + 1, stride=2, padding=1))

        self.final_conv_module = ConvModule(128, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        out = x
        intermediate_values = []
        for i, layer in enumerate(self.contract):
            out = layer(out)
        intermediate_values.append(out)
        if i != len(self.contract) - 1:
            out = F.max_pool2d(out, 2)

        for i, layer in enumerate(self.expand):
            out = self.deconvs[i](out)
            out = layer(intermediate_values[-(i + 2)], out)
        out = self.deconvs[-1](out)

        out = self.final_conv_module(torch.cat((intermediate_values[0], out), dim=1))
        out = self.final_conv(out)

        return out

class WNet(nn.Module):
    def __init__(self, n_modules=9, k=3):
        super().__init__()
        self.n_modules = n_modules
        self.k = k
        self.initialize_layers()

    def initialize_layers(self):
        self.encoder = UEncoder(n_modules=self.n_modules, k=self.k)
        self.decoder = UDecoder(n_modules=self.n_modules, k=self.k)

    def forward(self, x):
        segmentation = self.encoder(x)
        reconstruction = self.decoder(segmentation)

        return segmentation, reconstruction

    def dec_loss(self, x, reconstruction=None):
        if reconstruction is None:
            reconstruction = self.forward(x)[1]

        return torch.sum(torch.pow(x - reconstruction, 2))

    def enc_loss(self, x, sigma_i=10, sigma_x=4, r=5, segmentation=None, reconstruction=None):
        if segmentation is None or reconstruction is None:
            segmentation, reconstruction = self.forward(x)

        indices = x.indices()
        association = 0
        for k in range(self.k):
            A_k = segmentation[:, k, ...]
        window = indices[2]
        numerator = 0
        denominator = 0
        association += numerator / denominator

        loss = self.k - association

        return loss

    def _loss(self, x):
        segmentation, reconstruction = self.forward(x)
        dec_loss = self.dec_loss(x, reconstruction=reconstruction)
        enc_loss = self.enc_loss(x, segmentation=segmentation, reconstruction=reconstruction)

        return [dec_loss + enc_loss]

    def loss(self, x):
        return [self.dec_loss(x)]
