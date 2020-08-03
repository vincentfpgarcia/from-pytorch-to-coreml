# Conversion from ONNX to CoreML: Image edition

The model conversion described in [step 4](step4.md) works fine but has several drawbacks:


1. The CoreML model accepts input image of type `MLMultiArray`, which is not very common in iOS.

2. The considered CoreML model accepts an input image encoded on `float32` with values defined in [0,1]. Unfortunately, most of the time, images are encoded on `uint8` with values defined in [0,255]. A format conversion is consequently needed.

In this section, I describe a solution that fixes these problems:

- [Part 1](step5_part1.md): Improved model conversion
- [Part 2](step5_part2.md): Display CoreML model specifications
- [Part 3](step5_part3.md): Test CoreML model on one image