import torch
import coremltools as ct

from model import MyNet


# Create the model and load the weights
model = MyNet()
model.load_state_dict(torch.load('my_network.pth'))

# Create dummy input
dummy_input = torch.rand(1, 3, 32, 32)

# Trace the model
traced_model = torch.jit.trace(model, dummy_input)

# Create the input image type
input_image = ct.ImageType(name="my_input", shape=(1, 3, 32, 32), scale=1/255)

# Convert the model
coreml_model = ct.convert(traced_model, inputs=[input_image])

# Modify the output's name to "my_output" in the spec
spec = coreml_model.get_spec()
ct.utils.rename_feature(spec, "81", "my_output")

# Re-create the model from the updated spec
coreml_model_updated = ct.models.MLModel(spec)

# Save the CoreML model
coreml_model_updated.save('my_network_image_ct4.mlmodel')