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
