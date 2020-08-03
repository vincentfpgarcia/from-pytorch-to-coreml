import coremltools

# Load the CoreML model
model =  coremltools.models.MLModel('my_network_image.mlmodel')

# Display its specifications
print(model.visualize_spec)
