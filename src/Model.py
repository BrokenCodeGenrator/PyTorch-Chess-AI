import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, input_channels, num_filters):

        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, num_filters):

        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU()

    def forward(self, x):

        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu2(x)
        return x


class ValueHead(nn.Module):

    def __init__(self, input_channels):

        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU()
        
        self.fc1 = nn.Linear(64, 256)  # Adjust for dynamic input
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)  # Dropout for regularization
        
        self.fc2 = nn.Linear(256, 128)

        self.relu3 = nn.ReLU()
        
        
        self.fc3 = nn.Linear(128, 1)  # Final output: single value prediction
        self.tanh1 = nn.Tanh()  # Keep output in range [-1, 1]

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = x.view(x.size(0), -1)  # Dynamically flatten the tensor
        
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu3(x)


        x = self.fc3(x)
        x = self.tanh1(x)
        
        return x


class ChessNet(nn.Module):

    def __init__(self, num_blocks, num_filters):

        super().__init__()
        self.convBlock1 = ConvBlock(16+12, num_filters)
        self.residualBlocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_blocks)]
        )
        self.valueHead = ValueHead(num_filters)

    def forward(self, x, valueTarget=None):

        x = self.convBlock1(x)
        for block in self.residualBlocks:
            x = block(x)

        value = self.valueHead(x)

        if self.training and valueTarget is not None:
            valueLoss = (value - valueTarget).abs().mean()
            return valueLoss
        else:
            return value
