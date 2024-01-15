import torch
import torch.nn as nn


class ResidualBlockEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_channels)
        )
        if stride > 1:
            self.downsampler = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))
        else:
            self.downsampler = None

        self.activation = nn.ELU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsampler:
            residual = self.downsampler(x)
        out += residual
        out = self.activation(out)
        return out

class ResidualBlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=1):
        super(ResidualBlockDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=upsample, padding=1, output_padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(out_channels)
        )
        if upsample > 1:
            self.upsampler = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=upsample, output_padding=1))
        else:
            self.upsampler = None

        self.activation = nn.ELU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.upsampler:
            residual = self.upsampler(x)
        out += residual
        out = self.activation(out)
        return out

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = self.buildEncoder(latent_dim)
        self.decoder = self.buildDecoder(latent_dim)

    def buildEncoder(self, latent_dim):
        encoder = nn.Sequential(

            ResidualBlockEncoder(in_channels=2, out_channels=24, stride=2),

            ResidualBlockEncoder(in_channels=24, out_channels=48, stride=2),

            nn.ConstantPad2d((0, 1, 0, 0), 0),
            ResidualBlockEncoder(in_channels=48, out_channels=96, stride=2),

            nn.ConstantPad2d((0, 0, 0, 1), 0),
            ResidualBlockEncoder(in_channels=96, out_channels=192, stride=2),

            nn.ConstantPad2d((0, 1, 0, 0), 0),
            ResidualBlockEncoder(in_channels=192, out_channels=256, stride=2),

            nn.ConstantPad2d((0, 0, 0, 1), 0),
            ResidualBlockEncoder(in_channels=256, out_channels=256, stride=2),

            nn.Flatten(start_dim=1, end_dim=-1),

            nn.Linear(2560, 256),
            nn.ELU(),

            nn.Linear(256, latent_dim * 2),
        )
        return encoder


    def buildDecoder(self, latent_dim):
        decoder = nn.Sequential(

            nn.Linear(latent_dim, 256),
            nn.ELU(),

            nn.Linear(256, 256 * 5 * 2),
            nn.ELU(),

            nn.Unflatten(dim=1, unflattened_size=(256, 2, 5)),

            ResidualBlockDecoder(in_channels=256, out_channels=256, upsample=2),
            nn.ConstantPad2d((0, 0, 0, -1), 0),

            ResidualBlockDecoder(in_channels=256, out_channels=192, upsample=2),
            nn.ConstantPad2d((0, -1, 0, 0), 0),

            ResidualBlockDecoder(in_channels=192, out_channels=96, upsample=2),
            nn.ConstantPad2d((0, 0, 0, -1), 0),

            ResidualBlockDecoder(in_channels=96, out_channels=48, upsample=2),
            nn.ConstantPad2d((0, -1, 0, 0), 0),

            ResidualBlockDecoder(in_channels=48, out_channels=24, upsample=2),
            nn.ConstantPad2d((0, 0, 0, 0), 0),

            nn.ConvTranspose2d(in_channels=24, out_channels=2, kernel_size=3, stride=2, padding=1, output_padding=1),

        )
        return decoder

    def sample(self, mean, logvariance):

        std = torch.exp(0.5 * logvariance)
        epsilon = torch.rand_like(std)

        return mean + epsilon*std

    def forward(self, data):

        mean_logvariance = self.encoder(data)

        mean, logvariance = torch.chunk(mean_logvariance, 2, dim=1)

        z = self.sample(mean, logvariance)

        reconstruction = self.decoder(z)

        return reconstruction, mean, logvariance


def save_checkpoint(state, path_name):

    print('Saving checkpoint')

    torch.save(state, path_name)

    print('Saved checkpoint')


def load_checkpoint(model, path_name, optimizer=None):

    print('Loading checkpoint')
    print(path_name)

    checkpoint = torch.load(path_name)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    print('Loaded checkpoint')

'''Testing code'''
if __name__ == "__main__":
    from torchsummary import summary

    example = VAE(latent_dim=10)
    summary(example, input_size=(2, 88, 300))
    