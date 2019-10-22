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

# Build the network
model = MyNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

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

# Save the network weights
torch.save(model.state_dict(), 'my_network.pth')
