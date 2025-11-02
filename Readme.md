MNIST GAN API Deployment Guide

1. Build Docker Image

docker build -t mnist-gan-api .


2. Run Container

docker run -d -p 8000:8000 --name gan-server mnist-gan-api


3. Generate Images (PNG Format)

Access the following API endpoint to generate handwritten digit images:

http://localhost:8000/generate?num_samples=16&format_type=image%2Fpng


Parameters:

• num_samples: Number of images to generate (16 in the example)

• format_type: Image format (supports image/png and image/jpeg)

Usage Example:

# Test API with curl
curl -o generated_digits.png "http://localhost:8000/generate?num_samples=16&format_type=image%2Fpng"


4. Other Available Endpoints

• Health Check: http://localhost:8000/health

• API Documentation: http://localhost:8000/docs

```