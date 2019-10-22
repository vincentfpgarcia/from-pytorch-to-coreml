# Train and save

In this page, I will introduce how to:

- Write a simple PyTorch neural network
- Train and test the network
- Save the trained network on disk


## Neural network

Bellow is the model I'm going to use in this guide. As you can read, it's a combination of convolution layers (+ ReLu + max pooling) followed by several fully connected layers (+ ReLu).

I will not detail what each layer does. However, the following some information might be useful. For this network to work, the input array must be a 3 channels 32x32 array (see next section about CIFAR10). And the last fully connected layer outputs 10 values (for the 10 classes of CIFAR10).

```python
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 24, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(24 * 4 * 4, 192)
        self.fc2 = nn.Linear(192, 96)
        self.fc3 = nn.Linear(96, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 24 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
# Transformation to apply
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Training set
trainset = torchvision.datasets.CIFAR10(root='/datasets',
                                        train=True,
                                        download=True,
                                        transform=transform)

# Test set
testset = torchvision.datasets.CIFAR10(root='/datasets',
                                       train=False,
                                       download=True,
                                       transform=transform)

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
- The training and test accuracies increases over epochs (expected)
- The training accuracy is higher than the test accuracy (expected)
- The final test accuracy is 67.48 %, which is far superior compared to the 10% resulting from a random choice ;)

```
Training set size : 50000
Test set size     : 10000
Image size        : 32 x 32

Training
Epoch :  0, mini-batch : 1000, loss : 2.302
Epoch :  0, mini-batch : 2000, loss : 2.294
Epoch :  0, mini-batch : 3000, loss : 2.157
Epoch :  0, mini-batch : 4000, loss : 1.990
Epoch :  0, mini-batch : 5000, loss : 1.890
Epoch :  0, training accuracy =  31.30 %, test accuracy =  31.69 %
Epoch :  1, mini-batch : 1000, loss : 1.753
Epoch :  1, mini-batch : 2000, loss : 1.664
Epoch :  1, mini-batch : 3000, loss : 1.569
Epoch :  1, mini-batch : 4000, loss : 1.534
Epoch :  1, mini-batch : 5000, loss : 1.476
Epoch :  1, training accuracy =  47.14 %, test accuracy =  47.23 %
Epoch :  2, mini-batch : 1000, loss : 1.426
Epoch :  2, mini-batch : 2000, loss : 1.393
Epoch :  2, mini-batch : 3000, loss : 1.349
Epoch :  2, mini-batch : 4000, loss : 1.346
Epoch :  2, mini-batch : 5000, loss : 1.302
Epoch :  2, training accuracy =  55.30 %, test accuracy =  54.16 %
Epoch :  3, mini-batch : 1000, loss : 1.255
Epoch :  3, mini-batch : 2000, loss : 1.245
Epoch :  3, mini-batch : 3000, loss : 1.212
Epoch :  3, mini-batch : 4000, loss : 1.178
Epoch :  3, mini-batch : 5000, loss : 1.166
Epoch :  3, training accuracy =  59.12 %, test accuracy =  57.84 %
Epoch :  4, mini-batch : 1000, loss : 1.130
Epoch :  4, mini-batch : 2000, loss : 1.103
Epoch :  4, mini-batch : 3000, loss : 1.093
Epoch :  4, mini-batch : 4000, loss : 1.086
Epoch :  4, mini-batch : 5000, loss : 1.075
Epoch :  4, training accuracy =  64.75 %, test accuracy =  62.31 %
Epoch :  5, mini-batch : 1000, loss : 1.033
Epoch :  5, mini-batch : 2000, loss : 1.025
Epoch :  5, mini-batch : 3000, loss : 1.025
Epoch :  5, mini-batch : 4000, loss : 0.992
Epoch :  5, mini-batch : 5000, loss : 0.991
Epoch :  5, training accuracy =  67.58 %, test accuracy =  63.80 %
Epoch :  6, mini-batch : 1000, loss : 0.931
Epoch :  6, mini-batch : 2000, loss : 0.961
Epoch :  6, mini-batch : 3000, loss : 0.956
Epoch :  6, mini-batch : 4000, loss : 0.939
Epoch :  6, mini-batch : 5000, loss : 0.952
Epoch :  6, training accuracy =  70.29 %, test accuracy =  65.29 %
Epoch :  7, mini-batch : 1000, loss : 0.878
Epoch :  7, mini-batch : 2000, loss : 0.885
Epoch :  7, mini-batch : 3000, loss : 0.885
Epoch :  7, mini-batch : 4000, loss : 0.901
Epoch :  7, mini-batch : 5000, loss : 0.888
Epoch :  7, training accuracy =  72.02 %, test accuracy =  66.18 %
Epoch :  8, mini-batch : 1000, loss : 0.816
Epoch :  8, mini-batch : 2000, loss : 0.838
Epoch :  8, mini-batch : 3000, loss : 0.832
Epoch :  8, mini-batch : 4000, loss : 0.852
Epoch :  8, mini-batch : 5000, loss : 0.859
Epoch :  8, training accuracy =  73.76 %, test accuracy =  66.22 %
Epoch :  9, mini-batch : 1000, loss : 0.770
Epoch :  9, mini-batch : 2000, loss : 0.802
Epoch :  9, mini-batch : 3000, loss : 0.805
Epoch :  9, mini-batch : 4000, loss : 0.790
Epoch :  9, mini-batch : 5000, loss : 0.812
Epoch :  9, training accuracy =  75.81 %, test accuracy =  67.48 %
```