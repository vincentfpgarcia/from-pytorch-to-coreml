from onnx_coreml import convert

# Load the ONNX model as a CoreML model
model = convert(model='my_network.onnx')

# Save the CoreML model
model.save('my_network.mlmodel')
