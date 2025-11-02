# app/generator.py
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import os
import matplotlib.pyplot as plt

# 导入原始 model.py 中的 Generator 定义
from helper_lib.model import Generator


class GANGenerator:
    def __init__(self, model_path: str = "saved_models/generator_best.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.generator = Generator().to(self.device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
        self.generator.eval()

    def generate_images(self, num_samples: int = 16, latent_dim: int = 100) -> np.ndarray:
        """
        Generate fake images using the trained generator.
        Returns: NumPy array of shape (N, H, W), normalized to [0, 1]
        """
        with torch.no_grad():
            noise = torch.randn(num_samples, latent_dim).to(self.device)
            fake_images = self.generator(noise)
            fake_images = fake_images.cpu().numpy()
            # Shape: (N, 1, 28, 28) → Remove channel dim and scale to [0,1]
            fake_images = (fake_images + 1) / 2.0  # Denormalize from [-1,1] to [0,1]
            fake_images = np.clip(fake_images, 0, 1)
            return fake_images.squeeze(axis=1)  # (N, 28, 28)

    @staticmethod
    def images_to_base64(img_array: np.ndarray) -> list:
        """
        Convert NumPy image array to list of Base64-encoded PNG strings.
        """
        encoded_images = []
        for img in img_array:
            pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
            buf = BytesIO()
            pil_img.save(buf, format='PNG')
            encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
            encoded_images.append(encoded)
        return encoded_images

    @staticmethod
    def images_to_png_bytes(img_array: np.ndarray, cols=4):
        """
        Combine multiple images into a grid and return as PNG bytes.
        """
        num_samples = len(img_array)
        rows = (num_samples + cols - 1) // cols
        figure_width = cols * 2
        figure_height = rows * 2

        fig, axes = plt.subplots(rows, cols, figsize=(figure_width, figure_height))
        axes = axes.flatten() if num_samples > 1 else [axes]

        for i in range(num_samples):
            axes[i].imshow(img_array[i], cmap='gray')
            axes[i].axis('off')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(pad=0)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        return buf.read()
