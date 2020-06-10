# Test the CoreML model

Coremltools is a set of tools developed by Apple. Among other things, it can use a CoreML model to perform the inference directly in Python. In other words, it allows to test the model before deploying it on mobile.

The original code presented here can be found in the file [step5.py](step5.py). The first part of the code should be pretty familiar to you by now. First, it's the imports. Second, I access the CIFAR10 test set. Note that I will use mini-batches of 1 image since I will predict one image at the time.

```python
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
testset = torchvision.datasets.CIFAR10(root='~/datasets',
                                       train=False,
                                       download=True,
                                       transform=transform)

# Test set loader
testloader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=2)
```

Then, I load the CoreML model and I display its specifications.

```python
# Load the CoreML model and display its specifications
model =  coremltools.models.MLModel('my_network.mlmodel')
print(model.visualize_spec)
```

By executing these 2 lines of code, the print function displays the following information:

```
<bound method MLModel.visualize_spec of input {
  name: "my_input"
  shortDescription: "MultiArray of shape (1, 1, 3, 32, 32). The first and second dimensions correspond to sequence and batch size, respectively"
  type {
    multiArrayType {
      shape: 3
      shape: 32
      shape: 32
      dataType: FLOAT32
    }
  }
}
output {
  name: "my_output"
  shortDescription: "MultiArray of shape (1, 1, 10, 1, 1). The first and second dimensions correspond to sequence and batch size, respectively"
  type {
    multiArrayType {
      dataType: FLOAT32
    }
  }
}
>
```

One can see the model specifications with the input / output names, sizes and types. This is very useful to use a given model without knowing anything about it. One important information here is that the model input and output types are `multiArrayType`. In iOS, this corresponds to the [MLMultiArray](https://developer.apple.com/documentation/coreml/mlmultiarray) type. But in Python, since we don't have access to this type, we need to manipulate Numpy array, which is pretty convenient.

Last part of the code now. I go through all images of the CIFAR10 test set, convert the considered image into a Numpy array (now you understand why) of the expected size, use the model to predict the class and compare the result to the ground-truth. At the end I can deduce and print the model accuracy. See how I use `my_input` and `my_output` here?

```python
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

Using this code, the accuracy of the CoreML model is 67.51 %. I recall that the accuracy on the same test set was 67.48 % at training time using PyTorch. The (very small) difference is probably due to a difference in the operations implementation. At this point, I'm confident that the model I'm about to use in my iOS app will work just fine.
