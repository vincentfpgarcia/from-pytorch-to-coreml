# Test CoreML model on the test set

The last part consists in verifying the CoreML model on the CIFAR10 test set.

The original code presented here can be found in the file [step4_part4.py](step4_part4.py).

The first part of the code should be pretty familiar to you by now. First, it's the imports. Second, I access the CIFAR10 test set. Note that I will use mini-batches of 1 image since I will predict one image at the time.

```python
import coremltools
import torch
import torchvision
import torchvision.transforms as transforms


# Test set
testset = torchvision.datasets.CIFAR10(root='~/datasets',
                                       train=False,
                                       download=True,
                                       transform=transforms.ToTensor())

# Test set loader
testloader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=2)
```

Then, I load the CoreML model, go through all images of the CIFAR10 test set, convert the considered image into a Numpy array of the expected size, use the model to predict the class and compare the result to the ground-truth. At the end I can deduce and print the model accuracy.

```python
# Load the CoreML model
model =  coremltools.models.MLModel('my_network.mlmodel')

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
```

Using this code, the accuracy of my CoreML model is 71.82 %. I recall that the accuracy on the same test set was 71.83 % at training time using PyTorch. The (very small) difference is probably due to a difference in the operations implementation. At this point, I'm confident that the model I'm about to use in my iOS app will work just fine.
