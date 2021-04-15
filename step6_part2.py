import coremltools

# Load the CoreML model
model =  coremltools.models.MLModel('my_network_image_ct4.mlmodel')

# Display its specifications
print(model)
