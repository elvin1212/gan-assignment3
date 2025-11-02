import torch
import matplotlib.pyplot as plt
import numpy as np
from model import get_model  

def generate_samples(saved_gen_path, device, latent_dim=100, num_samples=16):

    generator, _ = get_model('GAN')  
    generator.load_state_dict(torch.load(saved_gen_path, map_location=device))
    generator = generator.to(device)
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        fake_images = generator(noise)
        fake_images = fake_images.cpu().numpy()
        
        cols = int(num_samples ** 0.5)
        rows = (num_samples + cols - 1) // cols
        
        plt.figure(figsize=(cols, rows))
        for i in range(num_samples):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(fake_images[i, 0], cmap='gray')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    generator.train()

# if __name__ == "__main__":
#     device = torch.device('cpu')
#     generate_samples('saved_models/generator_best.pth', device)
