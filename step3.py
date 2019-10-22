import torch

from model import MyNet


# Create the model and load the weights
model = MyNet()
model.load_state_dict(torch.load('my_network.pth'))

# Create dummy input
dummy_input = torch.rand(1, 3, 32, 32)

# Define input / output names
input_names = ["my_input"]
output_names = ["my_output"]

# Convert the PyTorch model to ONNX
torch.onnx.export(model,
                  dummy_input,
                  "my_network.onnx",
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)