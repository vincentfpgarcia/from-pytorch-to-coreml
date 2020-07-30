# Train and save

In this page, I will introduce how to:

- Write a simple PyTorch neural network
- Train and test the network
- Save the trained network on disk


## Neural network

Bellow is the model I'm going to use in this guide. As you can read, it's a combination of convolution layers (batch normalization + ReLu + max pooling) followed by several fully connected layers (dropout + ReLu). The original code can be found in the file [model.py](model.py).

I will not detail what each layer does. However, the following information might be useful. For this network to work, the input array must be a 3 channels 32x32 array (see next section about CIFAR10). And the last fully connected layer outputs 10 values (for the 10 classes of CIFAR10).

```python
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
```

## Training, testing and saving

This is the longest piece of code introduced in this guide. The code should be self explanatory. I split the code to give some additional information. The original code can be found in the file [step1.py](step1.py).

First, I import the necessary modules. You can see that I import the previously defined network. I also define here some parameters used in the rest of the code.

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from model import MyNet

# Parameters
batch_size = 10
nb_epochs = 10
learning_rate = 0.001
momentum = 0.9
```

I am going to use the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Torch and torchvision provide nice helpers to access popular image classification datasets. Using the following code, I simply create the training and the test sets.

```python
# Training set
trainset = torchvision.datasets.CIFAR10(root='~/datasets',
                                        train=True,
                                        download=True,
                                        transform=transforms.ToTensor())

# Test set
testset = torchvision.datasets.CIFAR10(root='~/datasets',
                                       train=False,
                                       download=True,
                                       transform=transforms.ToTensor())

# Training set loader
trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)

# Test set loader
testloader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)

# Access datasets properties
train_shape = trainset.data.shape
test_shape = testset.data.shape
train_nb = train_shape[0]
test_nb = test_shape[0]
height = train_shape[1]
width = train_shape[2]
classes = trainset.classes    
print('Training set size : %d' % train_nb)
print('Test set size     : %d' % test_nb)
print('Image size        : %d x %d\n' % (height, width))
```

The `torchvision.datasets.CIFAR10` module returns PIL images. The `DataLoader` module manipulates tensors or Numpy arrays. The transformation `transforms.ToTensor()` passed as parameter of `torchvision.datasets.CIFAR10` simply converts PIL images into tensors.

Now I instantiate my network. I also define the loss and the optimizer.

```python
# Build the network
model = MyNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
```

And now starts the training part. I print the loss value every 1000 mini-batches. I also print the accuracy on the training set and on the test set at the end of each epoch.

```python
# Training
print('Training')
for epoch in range(nb_epochs):

    # Set the model to training mode
    model.train()

    # Running loss container
    running_loss = 0.0

    # Iterate through mini-batches
    for i, data in enumerate(trainloader, 0):

        # Get the mini-batch data
        images, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Running loss update
        running_loss += loss.item()

        # Print the running loss every 1000 mini-batches
        if i % 1000 == 999:
            print('Epoch : %2d, mini-batch : %4d, loss : %.3f' % (epoch, i+1, running_loss / 1000))
            running_loss = 0.0

    # Set the model to evaluation mode
    model.eval()

    # Compute training set accuracy
    training_correct = 0
    training_total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            training_correct += (predicted == labels).sum().item()
            training_total += labels.size(0)
    training_accuracy = 100. * training_correct / training_total

    # Compute test set accuracy
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    test_accuracy = 100. * test_correct / test_total

    # Print the accuracies
    print('Epoch : %2d, training accuracy = %6.2f %%, test accuracy = %6.2f %%' % (epoch, training_accuracy, test_accuracy) )
```

And finally, I save the network weights on disk.

```python
# Save the network weights
torch.save(model.state_dict(), 'my_network.pth')
```

## Output analysis

Using this code, I get the following output. Some notes:

- The training and test sets have respectively 50000 and 10000 images
- The input image size is 32x32
- The loss decreases (most of the time) over mini-batches (expected)
- The training and test accuracie increases (most of the time) over epochs (expected)
- The training accuracy is higher than the test accuracy (expected)
- The final test accuracy is 71.83 %, which is far superior compared to the 10% resulting from a random choice ;)

```
Training set size : 50000
Test set size     : 10000
Image size        : 32 x 32

Training
Epoch :  0, mini-batch : 1000, loss : 1.973
Epoch :  0, mini-batch : 2000, loss : 1.611
Epoch :  0, mini-batch : 3000, loss : 1.493
Epoch :  0, mini-batch : 4000, loss : 1.418
Epoch :  0, mini-batch : 5000, loss : 1.348
Epoch :  0, training accuracy =  50.29 %, test accuracy =  49.72 %
Epoch :  1, mini-batch : 1000, loss : 1.292
Epoch :  1, mini-batch : 2000, loss : 1.249
Epoch :  1, mini-batch : 3000, loss : 1.218
Epoch :  1, mini-batch : 4000, loss : 1.205
Epoch :  1, mini-batch : 5000, loss : 1.162
Epoch :  1, training accuracy =  58.88 %, test accuracy =  57.40 %
Epoch :  2, mini-batch : 1000, loss : 1.131
Epoch :  2, mini-batch : 2000, loss : 1.116
Epoch :  2, mini-batch : 3000, loss : 1.079
Epoch :  2, mini-batch : 4000, loss : 1.081
Epoch :  2, mini-batch : 5000, loss : 1.070
Epoch :  2, training accuracy =  67.04 %, test accuracy =  64.95 %
Epoch :  3, mini-batch : 1000, loss : 1.011
Epoch :  3, mini-batch : 2000, loss : 1.017
Epoch :  3, mini-batch : 3000, loss : 1.023
Epoch :  3, mini-batch : 4000, loss : 1.009
Epoch :  3, mini-batch : 5000, loss : 1.000
Epoch :  3, training accuracy =  67.20 %, test accuracy =  64.53 %
Epoch :  4, mini-batch : 1000, loss : 0.948
Epoch :  4, mini-batch : 2000, loss : 0.968
Epoch :  4, mini-batch : 3000, loss : 0.957
Epoch :  4, mini-batch : 4000, loss : 0.958
Epoch :  4, mini-batch : 5000, loss : 0.947
Epoch :  4, training accuracy =  67.66 %, test accuracy =  64.58 %
Epoch :  5, mini-batch : 1000, loss : 0.917
Epoch :  5, mini-batch : 2000, loss : 0.901
Epoch :  5, mini-batch : 3000, loss : 0.892
Epoch :  5, mini-batch : 4000, loss : 0.901
Epoch :  5, mini-batch : 5000, loss : 0.906
Epoch :  5, training accuracy =  72.14 %, test accuracy =  68.57 %
Epoch :  6, mini-batch : 1000, loss : 0.858
Epoch :  6, mini-batch : 2000, loss : 0.860
Epoch :  6, mini-batch : 3000, loss : 0.879
Epoch :  6, mini-batch : 4000, loss : 0.861
Epoch :  6, mini-batch : 5000, loss : 0.871
Epoch :  6, training accuracy =  74.42 %, test accuracy =  69.60 %
Epoch :  7, mini-batch : 1000, loss : 0.824
Epoch :  7, mini-batch : 2000, loss : 0.833
Epoch :  7, mini-batch : 3000, loss : 0.820
Epoch :  7, mini-batch : 4000, loss : 0.825
Epoch :  7, mini-batch : 5000, loss : 0.819
Epoch :  7, training accuracy =  74.88 %, test accuracy =  69.87 %
Epoch :  8, mini-batch : 1000, loss : 0.771
Epoch :  8, mini-batch : 2000, loss : 0.792
Epoch :  8, mini-batch : 3000, loss : 0.804
Epoch :  8, mini-batch : 4000, loss : 0.787
Epoch :  8, mini-batch : 5000, loss : 0.819
Epoch :  8, training accuracy =  72.53 %, test accuracy =  67.09 %
Epoch :  9, mini-batch : 1000, loss : 0.767
Epoch :  9, mini-batch : 2000, loss : 0.788
Epoch :  9, mini-batch : 3000, loss : 0.769
Epoch :  9, mini-batch : 4000, loss : 0.775
Epoch :  9, mini-batch : 5000, loss : 0.788
Epoch :  9, training accuracy =  78.51 %, test accuracy =  71.83 %
```