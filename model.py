import torch
import torch.nn as nn


class VoxCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(VoxCNN, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*8*8*8, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
