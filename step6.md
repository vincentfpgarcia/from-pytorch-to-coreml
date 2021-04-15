# Conversion from PyTorch to CoreML using coremltools 4

The model conversion methods presented in [step 4](step4.md) and in [step 5](step5.md) are adapted to coremltools 3. The main drawback is the need to use a ONNX model as an intermediate step. With coremltools 4, we can convert directly a PyTorch model into a CoreML model.

In this section, I present how the conversion is done:

- [Part 1](step6_part1.md): Conversion from PyTorch to CoreML
- [Part 2](step6_part2.md): Display CoreML model specifications
- [Part 3](step6_part3.md): Test CoreML model on one image