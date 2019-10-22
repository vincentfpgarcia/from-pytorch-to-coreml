# Load and test

In the [previous script](step1.md), I presented how to train an artificial neural network and save it on disk. In this script I simply make sure that (1) I'm able to load the weights of the network and (2) the accuracy is unchanged.

Again, I split the code to provide some information. The original code can be found in the file [step2.py](step2.py).

The necessary imports, including my neural network `MyNet` of course.

```python
import torch
import torchvision
import torchvision.transforms as transforms

from model import MyNet
```

Again, I'm going to use here the CIFAR10 dataset through torchvision. But this time I will only use the test set.

```python
# Parameters
batch_size = 10

# Transformation to apply
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Test set
testset = torchvision.datasets.CIFAR10(root='/datasets',
                                       train=False,
                                       download=True,
                                       transform=transform)

# Test set loader
testloader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)
```

Next, well it's pretty self explanatory I think.

```python
# Create the model and load the weights
model = Net()
model.load_state_dict(torch.load('my_network.pth'))

# Set the model to evaluation mode
model.eval()
```

An finally I compute the accuracy:

```python
# Compute test set accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        predicted = torch.argmax(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
accuracy = 100. * correct / total

# Print the accuracy
print('Test accuracy = %6.2f %%' % accuracy)
```

Using this script, I obtain a 67.48 % accuracy on the test set. Good that's exactly what I had at training time on the same dataset.