import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential (

            # First convolutional layer
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolutional layer
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third convolutional layer
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential (

            # Dropout layer
            nn.Dropout(p=0.1),

            # First fully connected layer
            nn.Linear(in_features=24 * 4 * 4, out_features=192),
            nn.ReLU(inplace=True),

            # Second fully connected layer
            nn.Linear(in_features=192, out_features=96),
            nn.ReLU(inplace=True),

            # Third fully connected layer
            nn.Linear(in_features=96, out_features=10),
        )


    def forward(self, x):

        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten
        x = x.view(-1, 24 * 4 * 4)

        # Fully connected layers
        x = self.fc_layers(x)

        return x
