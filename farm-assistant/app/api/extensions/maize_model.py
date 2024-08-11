from torch import nn
import torch
from pydantic import BaseModel


class MaizeNet(nn.Module):
    def __init__(self, K) -> None:
        super(MaizeNet, self).__init__()

        self.conv_layers = nn.Sequential(
            # convolution 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # Convolution 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # Convolution 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # Convolution 4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            # Dropout layer
            nn.Dropout(0.5),
            # first fully connected layer
            nn.Linear(224*224, 1024),
            # Relu activation function
            nn.ReLU(),
            nn.Dropout(0.4),
            # Final output layer
            nn.Linear(1024, K),
        )

    def forward(self, output):
        # Convolution Layers
        out = self.conv_layers(output)

        # Flatten the layers
        out = out.view(-1, 224*224)

        # Fully connected Dense Layers
        out = self.dense_layers(out)

        return out
    
    
def load_model(model_path: str):
    """Load the pytorch model."""
    n_classes = 4
    maizenet = MaizeNet(n_classes)
    maizenet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu') ))
    return maizenet