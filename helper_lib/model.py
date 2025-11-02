import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim, 128 * 7 * 7)
        self.bn1 = nn.BatchNorm1d(128 * 7 * 7)
        self.relu = nn.ReLU(True)
        
        self.conv_transpose1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_transpose2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.linear(z)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(x.size(0), 128, 7, 7)  # Reshape
        
        x = self.conv_transpose1(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv_transpose2(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout2d(0.3)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(0.3)
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


def get_model(model_name):

    if model_name == 'GAN':
        generator = Generator()
        discriminator = Discriminator()
        return generator, discriminator
        
    else:
        raise ValueError(f"Model name '{model_name}' is not supported. "
                        f"Choose from: FCNN, CNN, EnhancedCNN, VAE, GAN")

