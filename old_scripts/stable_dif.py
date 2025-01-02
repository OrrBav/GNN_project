
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import os
# import numpy as np
# from torchvision.utils import save_image
# from datetime import datetime
# import json
# import time
# from tqdm.auto import tqdm

# class DoubleConv(nn.Module):
#     """
#     Double Convolution block used in the U-Net architecture.
#     This creates a more robust feature extraction process.
#     """
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class DiffusionModel(nn.Module):
#     """
#     A simplified diffusion model specifically for landscapes with hidden digits.
#     Uses a U-Net architecture with added channels for digit embedding.
#     """
#     def __init__(self, time_dim=256):
#         super().__init__()
#         # Time embedding
#         self.time_dim = time_dim
#         self.time_mlp = nn.Sequential(
#             nn.Linear(1, time_dim),
#             nn.GELU(),
#             nn.Linear(time_dim, time_dim)
#         )

#         # Digit embedding (0-9)
#         self.digit_embedding = nn.Embedding(10, 16)
        
#         # Down sampling path
#         self.inc = DoubleConv(3, 64)  # Input: RGB image
#         self.down1 = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(64, 128)
#         )
#         self.down2 = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(128, 256)
#         )

#         # Bottleneck with digit information
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(256 + 16, 256, 1),  # Add digit embedding channels
#             DoubleConv(256, 256)
#         )

#         # Up sampling path
#         self.up1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
#             DoubleConv(128, 128)
#         )
#         self.up2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
#             DoubleConv(64, 64)
#         )
        
#         self.outc = nn.Conv2d(64, 3, kernel_size=1)  # Output: RGB image

#     def forward(self, x, t, digit):
#         # Time embedding
#         # Convert t to float32 before processing
#         t = t.float()  # Convert time steps to float
#         t_emb = self.time_mlp(t.unsqueeze(-1))
        
#         # Digit embedding
#         d_emb = self.digit_embedding(digit)
#         d_emb = d_emb.view(d_emb.shape[0], -1, 1, 1)
#         d_emb = d_emb.expand(-1, -1, x.shape[2]//4, x.shape[3]//4)

#         # Downsampling
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)

#         # Inject digit information at the bottleneck
#         x3 = torch.cat([x3, d_emb], dim=1)
#         x3 = self.bottleneck(x3)

#         # Upsampling
#         x = self.up1(x3)
#         x = self.up2(x)
#         x = self.outc(x)
        
#         return x

# # class LandscapeDataset(Dataset):
# #     """
# #     Dataset class for loading landscape images and preparing them for training.
# #     """
# #     def __init__(self, img_dir, img_size=256):
# #         self.img_dir = img_dir
# #         self.img_size = img_size
# #         self.images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        
# #         self.transform = transforms.Compose([
# #             transforms.Resize((img_size, img_size)),
# #             transforms.ToTensor(),
# #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# #         ])

# #     def __len__(self):
# #         return len(self.images)

# #     def __getitem__(self, idx):
# #         img_path = os.path.join(self.img_dir, self.images[idx])
# #         image = Image.open(img_path).convert('RGB')
# #         image = self.transform(image)
        
# #         # Randomly assign a digit for training
# #         digit = torch.randint(0, 10, (1,))[0]
        
# #         return image, digit

# class LandscapeDataset(Dataset):
#     """
#     Enhanced Dataset class that includes comprehensive checks at each stage
#     of image loading and processing.
#     """
#     def __init__(self, img_dir, img_size=256):
#         self.img_dir = img_dir
#         self.img_size = img_size
        
#         # Verify directory exists
#         if not os.path.exists(img_dir):
#             raise RuntimeError(f"Directory not found: {img_dir}")
            
#         # Get list of image files with thorough checking
#         self.images = []
#         print("\n=== Dataset Initialization Checks ===")
#         print(f"Scanning directory: {img_dir}")
        
#         total_files = 0
#         image_files = 0
#         for filename in os.listdir(img_dir):
#             total_files += 1
#             if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 self.images.append(filename)
#                 image_files += 1
        
#         print(f"Total files found: {total_files}")
#         print(f"Valid image files: {image_files}")
        
#         if len(self.images) == 0:
#             raise RuntimeError("No valid image files found in directory!")
        
#         # Define transformations with size checks
#         self.transform = transforms.Compose([
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
        
#         # Validate first few images
#         self.validate_initial_images()

#     def validate_initial_images(self):
#         """Perform detailed validation of first few images in dataset."""
#         print("\n=== Initial Image Validation ===")
#         validation_count = min(3, len(self.images))
        
#         for idx in range(validation_count):
#             img_path = os.path.join(self.img_dir, self.images[idx])
#             print(f"\nChecking image {idx + 1}: {self.images[idx]}")
            
#             try:
#                 # Check if file exists
#                 if not os.path.exists(img_path):
#                     print(f"Error: File does not exist: {img_path}")
#                     continue
                
#                 # Check file size
#                 file_size = os.path.getsize(img_path) / 1024  # Size in KB
#                 print(f"File size: {file_size:.1f} KB")
                
#                 # Try to open and verify image
#                 with Image.open(img_path) as img:
#                     # Basic image properties
#                     print(f"Original size: {img.size}")
#                     print(f"Mode: {img.mode}")
#                     print(f"Format: {img.format}")
                    
#                     # Convert to RGB if needed
#                     if img.mode != 'RGB':
#                         print(f"Converting from {img.mode} to RGB")
#                         img = img.convert('RGB')
                    
#                     # Apply transformations and check results
#                     transformed_img = self.transform(img)
                    
#                     print("\nTransformed image properties:")
#                     print(f"Shape: {transformed_img.shape}")
#                     print(f"Value range: [{transformed_img.min():.2f}, {transformed_img.max():.2f}]")
#                     print(f"Mean: {transformed_img.mean():.2f}")
#                     print(f"Std: {transformed_img.std():.2f}")
                    
#                     # Save sample of processed image
#                     if idx == 0:
#                         save_image(transformed_img, f'sample_processed_image_{idx}.png')
#                         print(f"Saved sample processed image: sample_processed_image_{idx}.png")
                
#                 print("✓ Image validated successfully")
                
#             except Exception as e:
#                 print(f"Error validating image: {str(e)}")
#                 raise  # Re-raise the exception to stop initialization if validation fails

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         """Get an item with additional checks."""
#         img_path = os.path.join(self.img_dir, self.images[idx])
        
#         try:
#             # Load and process image
#             image = Image.open(img_path).convert('RGB')
#             image = self.transform(image)
            
#             # Verify transformed image
#             if not self.verify_transformed_image(image):
#                 print(f"Warning: Unusual values in transformed image {self.images[idx]}")
            
#             # Generate random digit
#             digit = torch.randint(0, 10, (1,))[0]
            
#             return image, digit
            
#         except Exception as e:
#             print(f"Error processing image {img_path}: {str(e)}")
#             raise

#     def verify_transformed_image(self, img_tensor):
#         """Verify that transformed image tensor has expected properties."""
#         expected_shape = (3, self.img_size, self.img_size)
#         if img_tensor.shape != expected_shape:
#             return False
            
#         if img_tensor.min() < -1.5 or img_tensor.max() > 1.5:
#             return False
            
#         return True

# def train_step(model, optimizer, images, digits, t, noise_scheduler, device):
#     """
#     Single training step for the diffusion model.
#     Returns the loss tensor directly, maintaining its computational graph.
#     """
#     model.train()
    
#     # Create noise and add it to images
#     noise = torch.randn_like(images)
#     noisy_images = noise_scheduler.add_noise(images, noise, t)

#     # Predict noise
#     noise_pred = model(noisy_images, t, digits)
    
#     # Calculate loss - keeping it as a tensor
#     loss = F.mse_loss(noise_pred, noise)
    
#     # Optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     # Important: Don't call .item() here
#     return loss  # Return the raw tensor

# class NoiseScheduler:
#     """
#     Simple linear noise scheduler for the diffusion process
#     """
#     def __init__(self, num_timesteps=1000, device=None):
#         self.num_timesteps = num_timesteps
#         self.device = device if device is not None else torch.device('cpu')
        
#         # Initialize beta schedule
#         self.beta = torch.linspace(1e-4, 0.02, num_timesteps).to(self.device)
#         self.alpha = (1 - self.beta).to(self.device)
#         self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(self.device)

#     def add_noise(self, x, noise, t):
#         """Add noise to image at timestep t"""
#         # Ensure t is on the correct device
#         t = t.to(self.device)
#         alpha_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
#         return torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise

#     @torch.no_grad()
#     def sample(self, model, n_samples, size, device, digit):
#         """Generate new images using the trained model"""
#         model.eval()
#         x = torch.randn(n_samples, 3, size, size).to(device)
        
#         for t in reversed(range(self.num_timesteps)):
#             t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
#             predicted_noise = model(x, t_batch, digit)
#             alpha = self.alpha[t]
#             alpha_bar = self.alpha_bar[t]
#             beta = self.beta[t]
            
#             if t > 0:
#                 noise = torch.randn_like(x)
#             else:
#                 noise = torch.zeros_like(x)
                
#             x = 1 / torch.sqrt(alpha) * (
#                 x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise
#             ) + torch.sqrt(beta) * noise
            
#         return x


# def save_generated_images(samples, epoch, digit, save_dir):
#     """
#     Save generated images in a organized directory structure
#     Args:
#         samples: Tensor of generated images
#         epoch: Current training epoch
#         digit: The digit that was embedded
#         save_dir: Base directory for saving images
#     """
#     # Denormalize the images from [-1, 1] to [0, 1]
#     samples = (samples + 1) / 2.0
    
#     # Create directory structure
#     epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
#     os.makedirs(epoch_dir, exist_ok=True)
    
#     # Save each sample
#     save_path = os.path.join(epoch_dir, f"digit_{digit}.png")
#     save_image(samples, save_path)
    
#     # Save metadata
#     metadata = {
#         "epoch": epoch,
#         "embedded_digit": int(digit),
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }
#     metadata_path = os.path.join(epoch_dir, f"digit_{digit}_metadata.json")
#     with open(metadata_path, 'w') as f:
#         json.dump(metadata, f, indent=4)

# def setup_training_directories(base_dir="diffusion_results"):
#     """
#     Create and return paths for organizing training outputs
#     """
#     # Create timestamp-based run directory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
#     # Create subdirectories
#     generated_images_dir = os.path.join(run_dir, "generated_images")
#     checkpoints_dir = os.path.join(run_dir, "checkpoints")
    
#     # Create all directories
#     for dir_path in [generated_images_dir, checkpoints_dir]:
#         os.makedirs(dir_path, exist_ok=True)
    
#     return run_dir, generated_images_dir, checkpoints_dir

# # Training configuration
# # def main():
# #     # Device setup
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"\nUsing device: {device}")
# #     if torch.cuda.is_available():
# #         print(f"GPU: {torch.cuda.get_device_name(0)}")
    
# #     # Model setup
# #     model = DiffusionModel().to(device)
# #     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# #     noise_scheduler = NoiseScheduler(device=device)
    
# #     # Dataset setup
# #     dataset = LandscapeDataset("/home/dsi/lynnmolga/Gan_project/landscapes")
# #     dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
# #     # Directory setup
# #     run_dir, generated_images_dir, checkpoints_dir = setup_training_directories()
# #     print(f"Saving results to: {run_dir}")
    
# #     # Configuration setup
# #     config = {
# #         "num_epochs": 100,
# #         "batch_size": 16,
# #         "learning_rate": 1e-4,
# #         "image_size": 256,
# #         "num_timesteps": noise_scheduler.num_timesteps
# #     }
    
# #     # Save configuration
# #     with open(os.path.join(run_dir, "training_config.json"), 'w') as f:
# #         json.dump(config, f, indent=4)

# #     num_epochs = config["num_epochs"]
# #     print("\nStarting training...")

# #     # Main training loop with tqdm for epochs
# #     for epoch in tqdm(range(num_epochs), desc="Training Progress", position=0):
# #         total_loss = 0
        
# #         # Inner loop with tqdm for batches
# #         for batch_idx, (images, digits) in tqdm(enumerate(dataloader), 
# #                                               desc=f"Epoch {epoch + 1}", 
# #                                               total=len(dataloader), 
# #                                               position=1, 
# #                                               leave=False):
# #             images = images.to(device)
# #             digits = digits.to(device)
# #             t = torch.randint(0, noise_scheduler.num_timesteps, (images.shape[0],), device=device)
            
# #             loss = train_step(model, optimizer, images, digits, t, noise_scheduler, device)
# #             total_loss += loss

# #         avg_loss = total_loss / len(dataloader)
        
# #         # Save checkpoint every 10 epochs
# #         if epoch % 10 == 0:
# #             checkpoint_path = os.path.join(checkpoints_dir, f"model_epoch_{epoch}.pt")
# #             torch.save({
# #                 'epoch': epoch,
# #                 'model_state_dict': model.state_dict(),
# #                 'optimizer_state_dict': optimizer.state_dict(),
# #                 'loss': avg_loss,
# #             }, checkpoint_path)
        
# #         # Generate and save sample images every 5 epochs
# #         if epoch % 5 == 0:
# #             model.eval()
# #             with torch.no_grad():
# #                 # Generate images with tqdm
# #                 for digit in tqdm(range(10), desc="Generating Images", position=1, leave=False):
# #                     samples = noise_scheduler.sample(
# #                         model, 
# #                         n_samples=1,
# #                         size=256,
# #                         device=device,
# #                         digit=torch.tensor([digit], device=device)
# #                     )
# #                     save_generated_images(samples, epoch, digit, generated_images_dir)
# #             model.train()

# #     print("\nTraining completed!")

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\nUsing device: {device}")
    
#     model = DiffusionModel().to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
#     noise_scheduler = NoiseScheduler(device=device)
    
#     # Dataset setup
#     dataset = LandscapeDataset("/home/dsi/lynnmolga/Gan_project/landscapes")
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)   
#     # Directory setup
#     run_dir, generated_images_dir, checkpoints_dir = setup_training_directories()
#     print(f"Saving results to: {run_dir}")
    
#     # Training configuration
#     config = {
#         "num_epochs": 100,
#         "batch_size": 16,
#         "learning_rate": 1e-4,
#         "image_size": 256,
#         "num_timesteps": noise_scheduler.num_timesteps
#     }
    
#     print("\nStarting training...")
    
#     for epoch in tqdm(range(config["num_epochs"]), desc="Training Progress"):
#         epoch_loss = 0.0
#         batch_count = 0
        
#         # Process each batch
#         for batch_idx, (images, digits) in enumerate(dataloader):
#             images = images.to(device)
#             digits = digits.to(device)
#             t = torch.randint(0, noise_scheduler.num_timesteps, (images.shape[0],), device=device)
            
#             # Get loss tensor and keep track of its type
#             loss = train_step(model, optimizer, images, digits, t, noise_scheduler, device)
#             # print(f"\nLoss type: {type(loss)}")  # Debugging print
#             # print(f"Is tensor? {isinstance(loss, torch.Tensor)}")  # Debugging print
            
#             # Only convert to float for logging
#             loss_value = loss.item()
#             epoch_loss += loss_value
#             batch_count += 1
            
#             if batch_idx % 10 == 0:
#                 print(f"\nEpoch {epoch+1}, Batch {batch_idx}")
#                 print(f"Current batch loss: {loss_value:.4f}")
#                 print(f"Average loss so far: {epoch_loss/batch_count:.4f}")
        
#         # Generate samples every 5 epochs
#         if epoch % 5 == 0:
#             model.eval()
#             with torch.no_grad():
#                 for digit in tqdm(range(10), desc="Generating samples"):
#                     samples = noise_scheduler.sample(
#                         model,
#                         n_samples=1,
#                         size=256,
#                         device=device,
#                         digit=torch.tensor([digit], device=device)
#                     )
#                     save_generated_images(samples, epoch, digit, generated_images_dir)
#             model.train()

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
import random
from pathlib import Path
import glob
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

class DigitMaskGenerator:
    """
    Enhanced mask generator that creates stronger patterns for visible digits in landscapes.
    """
    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size
        self.base_size = min(image_size[0], image_size[1])
        
        try:
            font_path = self._get_system_font()
            # Increase font size to make digits more prominent
            self.font = ImageFont.truetype(font_path, size=int(self.base_size * 0.8))
        except Exception as e:
            print(f"Could not load system font: {e}")
            print("Falling back to default font...")
            self.font = ImageFont.load_default()
    
    def _get_system_font(self):
        """Find a suitable system font, with platform-specific fallbacks"""
        possible_fonts = [
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Use bold font
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf"  # Prefer bold fonts
        ]
        
        for font_path in possible_fonts:
            if os.path.exists(font_path):
                return font_path
        raise FileNotFoundError("No suitable system font found")

    def generate_mask(self, digit):
        """
        Generate a stronger mask for a given digit with enhanced visibility
        """
        # Create initial canvas
        large_size = (self.base_size * 2, self.base_size * 2)
        large_image = Image.new('L', large_size, 0)
        large_draw = ImageDraw.Draw(large_image)
        
        # Convert digit to string and get its size
        digit_str = str(digit)
        bbox = large_draw.textbbox((0, 0), digit_str, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate center position
        x = (large_size[0] - text_width) // 2
        y = (large_size[1] - text_height) // 2
        
        # Draw the main digit with higher intensity
        large_draw.text((x, y), digit_str, fill=255, font=self.font)
        
        # Add a slight glow effect
        large_image = large_image.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Enhance contrast to make the digit more prominent
        large_image = ImageEnhance.Contrast(large_image).enhance(1.5)
        
        # Resize to target size with maintained intensity
        image = large_image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor
        tensor = transforms.ToTensor()(image)
        
        # Apply multiple processing steps to create a stronger mask
        # First, enhance the contrast
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        tensor = torch.pow(tensor, 0.7)  # Adjust gamma to make digits more prominent
        
        # Apply a smoother blur for better blending
        tensor = transforms.GaussianBlur(kernel_size=5, sigma=1.0)(tensor)
        
        # Normalize to ensure strong influence during diffusion
        tensor = tensor * 0.8 + 0.1  # Scale to range [0.1, 0.9] for stronger influence
        
        return tensor
    
class LandscapeDataset(Dataset):
    """
    Dataset class that works with only landscape images and generates digit masks
    """
    def __init__(self, landscape_paths, transform=None):
        self.landscape_paths = landscape_paths
        self.transform = transform
        self.mask_generator = DigitMaskGenerator()
        
    def __len__(self):
        return len(self.landscape_paths)
    
    def __getitem__(self, idx):
        # Load landscape
        landscape = Image.open(self.landscape_paths[idx]).convert('RGB')
        if self.transform:
            landscape = self.transform(landscape)
            
        # Generate random digit and its mask
        digit = random.randint(0, 9)
        mask = self.mask_generator.generate_mask(digit)
        
        return landscape, mask, digit

class DigitConditionedUNet(nn.Module):
    """
    U-Net architecture for landscape generation with digit conditioning.
    Includes proper time embedding handling to match feature dimensions.
    """
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(DigitConditionedUNet, self).__init__()
        
        # Calculate the bottleneck features for time embedding
        self.bottleneck_features = features * 16
        
        # Encoder path (downsampling)
        self.enc1 = self.conv_block(in_channels + 1, features)
        self.enc2 = self.conv_block(features, features * 2)
        self.enc3 = self.conv_block(features * 2, features * 4)
        self.enc4 = self.conv_block(features * 4, features * 8)
        
        # Bottleneck
        self.bottleneck = self.conv_block(features * 8, self.bottleneck_features)
        
        # Time embedding network adjusted to match bottleneck features
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.bottleneck_features),
            nn.SiLU(),
            nn.Linear(self.bottleneck_features, self.bottleneck_features)
        )
        
        # Decoder path (upsampling)
        self.up4 = nn.ConvTranspose2d(self.bottleneck_features, features * 8, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(features * 16, features * 8)  # *16 due to skip connection
        
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(features * 8, features * 4)   # *8 due to skip connection
        
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(features * 4, features * 2)   # *4 due to skip connection
        
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(features * 2, features)       # *2 due to skip connection
        
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def conv_block(self, in_ch, out_ch):
        """Double convolution block with batch normalization"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, digit_mask, t):
        """
        Forward pass with matched feature dimensions throughout.
        
        Args:
            x (torch.Tensor): Input image [B, 3, H, W]
            digit_mask (torch.Tensor): Digit mask [B, 1, H, W]
            t (torch.Tensor): Timestep [B]
        """
        # Combine input image and digit mask
        x = torch.cat([x, digit_mask], dim=1)
        
        # Encoder path with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Process bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Process time embedding to match bottleneck features
        t_emb = self.time_embedding(t.unsqueeze(1))           # [B, bottleneck_features]
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)            # [B, bottleneck_features, 1, 1]
        
        # Add time embedding to bottleneck (dimensions now match)
        bottleneck = bottleneck + t_emb.expand_as(bottleneck)
        
        # Decoder path with skip connections
        d4 = self.up4(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)
    
class DiffusionModel:
    def __init__(self, num_timesteps=1000, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Modified beta schedule for stronger conditioning
        self.beta = torch.linspace(1e-4, 0.015, num_timesteps).to(device)  # Reduced maximum beta
        self.alpha = (1 - self.beta).to(device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)
        
        # Precalculate values with adjusted weights for stronger conditioning
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha)
        self.sqrt_recip_alpha = torch.sqrt(1 / self.alpha)
        
        # Adjust posterior variance for better detail preservation
        self.posterior_variance = self.beta * (1 - self.alpha_bar.roll(1)) / (1 - self.alpha_bar)
        self.posterior_variance[0] = self.beta[0]
    
    def forward_diffusion(self, x, t):
        """Enhanced forward diffusion process with stronger conditioning"""
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x, device=x.device)
        
        # Modified noise addition for better pattern preservation
        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_x, noise
    
    def reverse_diffusion(self, model, x, digit_mask, t):
        """Enhanced reverse diffusion with stronger conditioning weight"""
        timestep = t.item()
        
        # Increase the influence of the digit mask
        x_with_mask = x + digit_mask * 0.3  # Increased conditioning strength
        
        # Predict noise using the enhanced input
        predicted_noise = model(x_with_mask, digit_mask, t.float() / self.num_timesteps)
        
        alpha = self.alpha[timestep]
        alpha_bar = self.alpha_bar[timestep]
        beta = self.beta[timestep]
        
        # Modified denoising process with stronger pattern preservation
        x_0_pred = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise)
        
        if timestep > 0:
            noise = torch.randn_like(x, device=x.device)
            variance = torch.sqrt(self.posterior_variance[timestep])
            # Reduce noise influence in later steps
            noise_scale = min(1.0, timestep / (self.num_timesteps * 0.75))
            x_prev = x_0_pred + variance * noise * noise_scale
        else:
            x_prev = x_0_pred
        
        return x_prev
    
    def reverse_diffusion_loop(self, model, x, digit_mask):
        """
        Complete reverse diffusion process from noise to image.
        
        Args:
            model: The UNet model
            x (torch.Tensor): Initial noise
            digit_mask (torch.Tensor): Digit mask for conditioning
        """
        model.eval()
        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                # Create timestep tensor on correct device
                timestep = torch.tensor([t], device=x.device)
                
                # Single reverse step
                x = self.reverse_diffusion(model, x, digit_mask, timestep)
                
                # Optional: Add progress tracking
                if t % 100 == 0:
                    print(f"Denoising step {self.num_timesteps - t}/{self.num_timesteps}")
        
        return x


def train_model(model, diffusion, dataloader, num_epochs, device):
    """
    Comprehensive training function that handles model training and periodic image generation.
    This function trains the model while saving sample images every epoch to visualize progress.
    
    Args:
        model: The DigitConditionedUNet model to train
        diffusion: The DiffusionModel for noise addition/removal
        dataloader: DataLoader containing the training data
        num_epochs: Total number of training epochs
        device: Device to run the training on (cuda/cpu)
    """
    # Set up directory for saving progress images
    progress_dir = os.path.join('Gan_project', 'training_progress')
    os.makedirs(progress_dir, exist_ok=True)
    
    # Initialize the model for training
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Set up optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    total_batches = len(dataloader)
    
    # Main training loop
    for epoch in range(num_epochs):
        # Track loss for this epoch
        epoch_loss = 0.0
        
        # Process each batch
        for batch_idx, (landscape, mask, digit) in enumerate(dataloader):
            try:
                # Move data to the correct device
                landscape = landscape.to(device)
                mask = mask.to(device)
                
                # Generate random timesteps for diffusion
                t = torch.randint(0, diffusion.num_timesteps, (landscape.shape[0],), 
                                device=device)
                
                # Forward diffusion process
                noisy_landscape, noise = diffusion.forward_diffusion(landscape, t)
                
                # Predict noise and calculate loss
                timestep_values = t.float() / diffusion.num_timesteps
                noise_pred = model(noisy_landscape, mask, timestep_values)
                loss = criterion(noise_pred, noise)
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accumulate loss for this epoch
                epoch_loss += loss.item()
                
                # Print progress every 50 batches
                if batch_idx % 50 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Batch [{batch_idx}/{total_batches}], "
                          f"Loss: {loss.item():.4f}")
                    
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx}:")
                print(f"Landscape device: {landscape.device}")
                print(f"Mask device: {mask.device}")
                print(f"Model device: {next(model.parameters()).device}")
                print(f"Error message: {str(e)}")
                raise e
        
        # Calculate and print average loss for this epoch
        avg_epoch_loss = epoch_loss / total_batches
        print(f"\nEpoch [{epoch+1}/{num_epochs}] completed, "
              f"Average Loss: {avg_epoch_loss:.4f}")
        
        # Generate and save sample images every epoch
        print(f"\nGenerating sample images for epoch {epoch + 1}...")
        
        # Switch to evaluation mode for generation
        model.eval()
        with torch.no_grad():
            # Generate images for digits 0-9
            for digit in range(10):
                try:
                    # Generate the image
                    generated = generate_landscape_with_digit(
                        model, diffusion, digit, device, img_size=(256, 256)
                    )
                    
                    # Convert to PIL image for saving
                    img = transforms.ToPILImage()(generated.squeeze(0).cpu())
                    
                    # Create descriptive filename
                    filename = f'epoch_{epoch+1:03d}digit{digit}.png'
                    save_path = os.path.join(progress_dir, filename)
                    
                    # Save the image
                    img.save(save_path)
                    print(f"Saved sample with digit {digit} for epoch {epoch + 1}")
                
                except Exception as e:
                    print(f"Error generating image for digit {digit} at epoch {epoch + 1}: {str(e)}")
                    continue
        
        # Switch back to training mode
        model.train()
        
        # Save model checkpoint every epoch
        checkpoint_path = os.path.join('Gan_project', f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        print(f"Saved model checkpoint for epoch {epoch + 1}\n")

def generate_landscape_with_digit(model, diffusion, digit, device, img_size=(256, 256)):
    """
    Generates a landscape image with a hidden digit using the trained model.
    This function handles the complete generation process from creating the digit mask
    to running the reverse diffusion process.
    
    Args:
        model: The trained DigitConditionedUNet model
        diffusion: The DiffusionModel instance
        digit: The digit to hide in the landscape (0-9)
        device: The device to run generation on (cuda/cpu)
        img_size: Tuple of (height, width) for the output image
    
    Returns:
        torch.Tensor: The generated image tensor, normalized to [0, 1]
    """
    model.eval()  # Set to evaluation mode
    
    with torch.no_grad():
        # Create mask for the specified digit
        mask_generator = DigitMaskGenerator(img_size)
        digit_mask = mask_generator.generate_mask(digit)
        digit_mask = digit_mask.unsqueeze(0).to(device)  # Add batch dimension
        
        # Start from random noise
        x = torch.randn((1, 3, *img_size), device=device)
        
        # Run the reverse diffusion process
        for t in reversed(range(diffusion.num_timesteps)):
            # Create timestep tensor
            timesteps = torch.tensor([t], device=device)
            
            # Generate one step
            # Assuming reverse_diffusion method exists in your DiffusionModel class
            x = diffusion.reverse_diffusion(
                model,
                x,
                digit_mask,
                timesteps
            )
        
        # Normalize the output image to [0, 1] range
        x = (x.clamp(-1, 1) + 1) / 2
        
    return x

def save_generated_image(generated_tensor, output_path):
    """
    Save the generated tensor as an image file.

    Args:
        generated_tensor (torch.Tensor): The tensor representing the image, should be in [0, 1]
        output_path (str): The path where to save the image
    """
    # Convert tensor to PIL Image
    pil_image = transforms.ToPILImage()(generated_tensor.squeeze(0).cpu())
    
    # Save the image
    pil_image.save(output_path)

def setup_training(landscape_dir, batch_size=16, image_size=(256, 256)):
    """
    Sets up the training environment and data loading pipeline for landscape images.
    
    Args:
        landscape_dir (str): Directory containing landscape images
        batch_size (int): Number of images to process in each training batch
        image_size (tuple): Target size for the images (width, height)
    
    Returns:
        DataLoader: PyTorch DataLoader configured for training
        
    Raises:
        FileNotFoundError: If no compatible images are found in the directory
        RuntimeError: If there are issues loading or processing the images
    """
    # First, let's verify the directory exists
    if not os.path.exists(landscape_dir):
        raise FileNotFoundError(f"Directory not found: {landscape_dir}")
        
    # Set up image transformations for training
    # These transforms help improve model training by:
    # 1. Resizing images to a consistent size
    # 2. Adding horizontal flips for data augmentation
    # 3. Converting to tensors and normalizing pixel values
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),  # Ensure exact dimensions
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Collect all image paths, supporting both .jpg and .jpeg extensions
    landscape_paths = []
    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
        pattern = os.path.join(landscape_dir, f'*{ext}')
        landscape_paths.extend(glob.glob(pattern))
    
    # Verify we found some images
    if not landscape_paths:
        raise FileNotFoundError(
            f"No JPEG images found in {landscape_dir}. "
            "Please ensure your images are in JPEG format and the path is correct."
        )
    
    print(f"Found {len(landscape_paths)} images for training")
    
    try:
        # Create the dataset
        dataset = LandscapeDataset(landscape_paths, transform=transform)
        
        # Configure the dataloader with appropriate settings
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),  # Speeds up data transfer to GPU
            drop_last=True  # Ensures all batches are the same size
        )
        
        # Verify we can load at least one batch
        test_batch = next(iter(dataloader))
        print(f"Successfully verified data loading. Batch shape: {test_batch[0].shape}")
        
        return dataloader
        
    except Exception as e:
        raise RuntimeError(f"Error setting up data loading: {str(e)}")
    

def main():
    """
    Main execution function for training and generating landscapes with hidden digits.
    Handles training, model saving, and example generation with comprehensive error checking.
    """
    # Configuration settings for the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training hyperparameters - feel free to adjust these
    image_size = (256, 256)  # Size of input/output images
    batch_size = 16          # Number of images processed at once
    num_epochs = 100         # Number of complete passes through the dataset
    
    # Path to your landscape images
    landscape_dir = "/home/dsi/orrbavly/.cache/kagglehub/datasets/arnaud58/landscape-pictures/versions/2/*.jpg"
    
    # Create output directory for generated images if it doesn't exist
    output_dir = "/home/dsi/orrbavly/GNN_project/old_scripts/lynn"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for landscape images in: {landscape_dir}")
    
    # Create model and move to device
    model = DigitConditionedUNet().to(device)
    diffusion = DiffusionModel(num_timesteps=1000)
    
    # Prepare the training data
    try:
        # Remove the asterisk from the path for the setup_training function
        base_landscape_dir = os.path.dirname(landscape_dir)
        dataloader = setup_training(base_landscape_dir, batch_size, image_size)
        print(f"Successfully loaded {len(dataloader.dataset)} landscape images")
        print(f"Will train for {num_epochs} epochs with batch size {batch_size}")
    except FileNotFoundError as e:
        print(f"Error: Could not find landscape images at {landscape_dir}")
        print("Please check that the path is correct and contains .jpg files")
        return
    except Exception as e:
        print(f"Unexpected error setting up training: {e}")
        return
    
    # Training loop
    print("Starting training...")
    try:
        train_model(model, diffusion, dataloader, num_epochs, device)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # # Save the trained model in the project directory
    # model_path = os.path.join('Gan_project', 'landscape_digit_model.pth')
    # try:
    #     torch.save(model.state_dict(), model_path)
    #     print(f"Model saved successfully to {model_path}")
    # except Exception as e:
    #     print(f"Error saving model: {e}")
    #     return
    
       # Generate example outputs
    print("Generating example landscapes with hidden digits...")
    for digit in range(10):
        try:
            generated = generate_landscape_with_digit(model, diffusion, digit, device, image_size)
            # Save the generated image in the output directory
            output_path = os.path.join(output_dir, f'landscape_with_{digit}.png')
            save_generated_image(generated, output_path)
            print(f"Generated landscape with hidden digit {digit} saved to {output_path}")
        except Exception as e:
            print(f"Error generating landscape with digit {digit}: {e}")
            continue  # Continue with next digit even if one fails

if __name__ == '__main__':
    main()