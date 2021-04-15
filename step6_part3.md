# Test CoreML model on one image

As in [step 4](step4_part3.md) and [step 5](step5_part3.md), the purpose here is not to validate the model but to explain how to use the CoreML model and coremltools to perform an inference in Python.

The original code presented here can be found in the file [step6_part3.py](step6_part3.py).

```python
import coremltools
import numpy as np
from PIL import Image


# CIFAR10 classes
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the test image
image = Image.open('dog.png')

# Load the CoreML model
model =  coremltools.models.MLModel('my_network_image_ct4.mlmodel')

# Prediction vector as a numpy array
pred = model.predict({'my_input': image})
pred = pred['my_output']
pred = pred.squeeze()

# Display the most probable class
idx = pred.argmax()
print('Predicted class : %d (%s)' % (idx, cifar10_classes[idx]))
```

The code is identical to the one of [step 5](step5_part3.py). Using this modified model, the most probable class is:

```
Predicted class : 5 (dog)
```
