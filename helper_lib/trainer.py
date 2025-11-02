import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
from model import get_model  
import matplotlib.pyplot as plt
import numpy as np
import os

def train_gan():
    torch.manual_seed(45)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

    generator, discriminator = get_model('GAN')  
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    transform = Compose([
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])
    dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=30)
    
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    best_loss = float('inf')
    best_epoch = 0
    os.makedirs('saved_models', exist_ok=True)
    
    num_epochs = 1000
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            d_optimizer.zero_grad()
            label_real = torch.ones(batch_size, 1).to(device)
            label_fake = torch.zeros(batch_size, 1).to(device)
            
            output_real = discriminator(real_images)
            d_loss_real = criterion(output_real, label_real)
            
            noise = torch.randn(batch_size, 100).to(device)  # latent_dim=100
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(output_fake, label_fake)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            g_optimizer.zero_grad()
            output_fake = discriminator(fake_images)
            g_loss = criterion(output_fake, label_real)
            
            g_loss.backward()
            g_optimizer.step()
            
            if i % 1000 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'd_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f}')
        
        current_loss = d_loss_real.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            torch.save(generator.state_dict(), 'saved_models/generator_best.pth')
            torch.save(discriminator.state_dict(), 'saved_models/discriminator_best.pth')
            print(50*"=")
            print(f"New best model saved at epoch {epoch+1} with d_loss_real: {current_loss:.4f}")
            print(50*"=")
        
        if (epoch + 1) % 10 == 0:
            save_fake_images(generator, epoch + 1, device)

    print(f"\n🏆 Training completed.")
    print(f"Best model saved at epoch {best_epoch + 1} with d_loss_real: {best_loss:.4f}")

def save_fake_images(generator, epoch, device, num_samples=16):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, 100).to(device)
        fake_images = generator(noise)
        fake_images = fake_images.cpu().numpy()
        
        plt.figure(figsize=(4, 4))
        for i in range(num_samples):
            plt.subplot(4, 4, i + 1)
            plt.imshow(fake_images[i, 0], cmap='gray')
            plt.axis('off')
        
        plt.savefig(f'fake_images_epoch_{epoch}.png')
        plt.close()
    generator.train()

if __name__ == '__main__':
    train_gan()
