from onnx_coreml import convert

# Load the ONNX model as a CoreML model
model = convert(
    model='my_network.onnx',
    image_input_names=['my_input'],
    preprocessing_args={'image_scale': 1./255.},
    minimum_ios_deployment_target='13')

# Save the CoreML model
model.save('my_network_image.mlmodel')
