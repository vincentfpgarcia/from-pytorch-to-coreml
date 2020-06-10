import torch
import torchvision
import torchvision.transforms as transforms

from model import MyNet


# Parameters
batch_size = 10

# Transformation to apply
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Test set
testset = torchvision.datasets.CIFAR10(root='~/datasets',
                                       train=False,
                                       download=True,
                                       transform=transform)

# Test set loader
testloader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)

# Create the model and load the weights
model = MyNet()
model.load_state_dict(torch.load('my_network.pth'))

# Set the model to evaluation mode
model.eval()

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
