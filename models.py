import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetBlock(nn.Module):
    """U-Net building block with skip connections"""
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False, use_bn=True):
        super(UNetBlock, self).__init__()
        self.down = down
        
        if down:
            # Encoder (downsampling)
            self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        else:
            # Decoder (upsampling)
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.dropout = nn.Dropout(0.5) if use_dropout else None
        self.activation = nn.LeakyReLU(0.2) if down else nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class Generator(nn.Module):
    """
    U-Net Generator for Pix2Pix
    Input: Sketch image (3 channels)
    Output: Real face image (3 channels)
    """
    def __init__(self, input_channels=3, output_channels=3, features=64):
        super(Generator, self).__init__()
        
        # Encoder (Downsampling)
        self.e1 = nn.Conv2d(input_channels, features, 4, 2, 1)  # 256 -> 128
        self.e2 = UNetBlock(features, features*2, down=True)      # 128 -> 64
        self.e3 = UNetBlock(features*2, features*4, down=True)   # 64 -> 32
        self.e4 = UNetBlock(features*4, features*8, down=True)   # 32 -> 16
        self.e5 = UNetBlock(features*8, features*8, down=True)   # 16 -> 8
        self.e6 = UNetBlock(features*8, features*8, down=True)   # 8 -> 4
        self.e7 = UNetBlock(features*8, features*8, down=True)   # 4 -> 2
        self.e8 = UNetBlock(features*8, features*8, down=True, use_bn=False)  # 2 -> 1
        
        # Decoder (Upsampling)
        self.d1 = UNetBlock(features*8, features*8, down=False, use_dropout=True)  # 1 -> 2
        self.d2 = UNetBlock(features*16, features*8, down=False, use_dropout=True) # 2 -> 4
        self.d3 = UNetBlock(features*16, features*8, down=False, use_dropout=True) # 4 -> 8
        self.d4 = UNetBlock(features*16, features*8, down=False)                   # 8 -> 16
        self.d5 = UNetBlock(features*16, features*4, down=False)                   # 16 -> 32
        self.d6 = UNetBlock(features*8, features*2, down=False)                    # 32 -> 64
        self.d7 = UNetBlock(features*4, features, down=False)                      # 64 -> 128
        
        # Final layer
        self.final = nn.ConvTranspose2d(features*2, output_channels, 4, 2, 1)     # 128 -> 256
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        
        # Decoder with skip connections
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1))
        d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1))
        d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))
        
        # Final output
        output = self.final(torch.cat([d7, e1], dim=1))
        return self.tanh(output)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix
    Input: Concatenated sketch and real/fake image (6 channels)
    Output: Patch-wise real/fake classification
    """
    def __init__(self, input_channels=6, features=64):
        super(Discriminator, self).__init__()
        
        # PatchGAN layers
        self.conv1 = nn.Conv2d(input_channels, features, 4, 2, 1)
        self.conv2 = self._discriminator_block(features, features*2)
        self.conv3 = self._discriminator_block(features*2, features*4)
        self.conv4 = self._discriminator_block(features*4, features*8, stride=1)
        self.conv5 = nn.Conv2d(features*8, 1, 4, 1, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def _discriminator_block(self, in_channels, out_channels, stride=2):
        """Create a discriminator block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, sketch, image):
        # Concatenate sketch and image
        x = torch.cat([sketch, image], dim=1)
        
        # Apply discriminator layers
        x = self.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return self.sigmoid(x)


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_device():
    """
    Get the best available device for training
    Supports CUDA (NVIDIA), ROCm (AMD), DirectML (AMD), and CPU
    """
    # Check for CUDA (NVIDIA)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name()}")
        return device
    
    # Check for ROCm (AMD)
    try:
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            device = torch.device('cuda')  # ROCm uses cuda syntax
            print("Using AMD GPU with ROCm")
            return device
    except:
        pass
    
    # Check for DirectML (AMD/Intel)
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            print("Using AMD/Intel GPU with DirectML")
            return device
    except ImportError:
        pass
    
    # Fallback to CPU
    device = torch.device('cpu')
    print("Using CPU")
    return device


def create_generators_and_discriminator(device=None):
    """
    Create and initialize Generator and Discriminator
    
    Args:
        device: torch device (auto-detect if None)
        
    Returns:
        tuple: (generator, discriminator)
    """
    if device is None:
        device = get_device()
    
    # Create models
    generator = Generator(input_channels=3, output_channels=3, features=64).to(device)
    discriminator = Discriminator(input_channels=6, features=64).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    return generator, discriminator


def test_models():
    """Test model architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    generator, discriminator = create_generators_and_discriminator(device)
    
    # Test input
    batch_size = 4
    sketch = torch.randn(batch_size, 3, 256, 256).to(device)
    real_image = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print("\n=== Model Architecture Test ===")
    
    # Test Generator
    print(f"Input sketch shape: {sketch.shape}")
    fake_image = generator(sketch)
    print(f"Generated image shape: {fake_image.shape}")
    print(f"Generated image range: {fake_image.min():.3f} to {fake_image.max():.3f}")
    
    # Test Discriminator
    disc_real = discriminator(sketch, real_image)
    disc_fake = discriminator(sketch, fake_image.detach())
    print(f"Discriminator real output shape: {disc_real.shape}")
    print(f"Discriminator fake output shape: {disc_fake.shape}")
    print(f"Discriminator real range: {disc_real.min():.3f} to {disc_real.max():.3f}")
    print(f"Discriminator fake range: {disc_fake.min():.3f} to {disc_fake.max():.3f}")
    
    # Model parameters
    gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    
    print(f"\n=== Model Statistics ===")
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Total parameters: {gen_params + disc_params:,}")
    
    # Memory usage estimate
    gen_memory = gen_params * 4 / 1024 / 1024  # 4 bytes per float32, convert to MB
    disc_memory = disc_params * 4 / 1024 / 1024
    
    print(f"\n=== Memory Usage (approx) ===")
    print(f"Generator: {gen_memory:.1f} MB")
    print(f"Discriminator: {disc_memory:.1f} MB")
    print(f"Total: {gen_memory + disc_memory:.1f} MB")
    
    print("\n✅ Model architecture test completed successfully!")


if __name__ == "__main__":
    test_models()
