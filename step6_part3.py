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
