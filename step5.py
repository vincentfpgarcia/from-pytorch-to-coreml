import coremltools
import torch
import torchvision
import torchvision.transforms as transforms


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
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=2)


# Load the CoreML model and display its specifications
model =  coremltools.models.MLModel('my_network.mlmodel')
print(model.visualize_spec)

# Compute the accuracy on the test set
correct = 0
total = 0
for data in testloader:

        # Access the current test image and corresponding label
        image, label = data

        # The current image is a PyTorch Tensor of size [1, 3, 32, 32]
        # We need to convert it to a Numpy array of size [3, 32, 32]
        image = image.numpy().squeeze()

        # Convert the label as a number
        label = label.numpy()[0]

        # Prediction as a numpy array
        pred = model.predict({'my_input': image})
        pred = pred['my_output']
        pred = pred.squeeze()

        # Deduce the predicted label by locating the highest value in the prediction vector
        pred_label = pred.argmax()

        # Update values for accuracy computation
        if pred_label == label:
            correct += 1
        total += 1

# Print the accuracy
accuracy = 100. * correct / total
print('Test accuracy = %6.2f %%' % accuracy)
