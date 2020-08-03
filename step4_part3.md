# Test CoreML model on one image

In the following code, I load the converted CoreML model and test it on one image from the CIFAR10 dataset. The purpose here is not to validate the model but to explain how to use the CoreML model and coremltools to perform an inference in Python.

The original code presented here can be found in the file [step4_part3.py](step4_part3.py).

```python
import coremltools
import numpy as np
from PIL import Image


# CIFAR10 classes
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the test image
image = Image.open('dog.png')

# Convert the image to a Numpy array with values in [0,1]
image = np.array(image)
image = image.astype(np.float32)
image = image / 255.

# Reorder the image axis so that the image size is 3 x 32 x 32 (channels x height x width)
image = image.transpose(2, 0, 1)

# Load the CoreML model
model =  coremltools.models.MLModel('my_network.mlmodel')

# Prediction vector as a numpy array
pred = model.predict({'my_input': image})
pred = pred['my_output']
pred = pred.squeeze()

# Display the most probable class
idx = pred.argmax()
print('Predicted class : %d (%s)' % (idx, cifar10_classes[idx]))
```

At the beginning of the script, I load the `dog.png` image using PIL. Since the CoreML model can only manipulate Numpy arrays, I convert the PIL image into a Numpy array where values are defined on [0,1] and where the image dimension is 3 x 32 x 32.

Then, I load the model and process the image using the `predict` function. See how I use the layer names `my_input` and `my_output`?

Finally, I display the most probable class considering the input image:

```
Predicted class : 5 (dog)
```

It seems that my CoreML model is able to recognize a dog. ^^