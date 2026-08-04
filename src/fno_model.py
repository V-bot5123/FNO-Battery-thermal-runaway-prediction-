import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # We only keep the lowest frequencies to capture main physics
        self.modes1 = modes1
        self.modes2 = modes2
        
        # R is a weight tensor that holds Complex Numbers
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        nn.init.xavier_normal_(self.weights1)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # 1. Fourier Transform (Space -> Frequency)
        x_ft = torch.fft.rfft2(x)
        
        # 2. Empty tensor for output frequencies
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), 
                             x_ft.size(-1), device=x.device, dtype=torch.cfloat)
        
        # 3. Multiply input frequencies by learnable complex weights
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", 
                         x_ft[:, :, :self.modes1, :self.modes2], 
                         self.weights1)
        
        # 4. Inverse Fourier Transform (Frequency -> Space)
        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x_out


class FNO2d(nn.Module):
    """
    The full Fourier Neural Operator
    """
    def __init__(self, modes=12, width=32):
        super(FNO2d, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        
        # Lifts the input (1 channel for temperature) to higher dimensional space
        self.fc0 = nn.Linear(1, self.width) 
        
        # The Spectral Convolution Layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # Standard Linear bypass layers
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)

        # Projects back down to 1 channel (temperature)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x shape: (batch, size_x, size_y) -> add channel dimension
        x = x.unsqueeze(-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        
        # Project back
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        # Remove channel dimension
        x = x.squeeze(-1)
        return x
