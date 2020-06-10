# Motivation

I present here a simple guide that explains the steps needed from training a simple PyTorch image classifier to converting the trained neural network into a CoreML model ready for production. I've spent days crawling Internet blogs, forums and official documentations to gather the little knowledge presented in these pages. The true motivation of this repo is to prevent me to forget everything I know about this particular subject. If this guide helps someone else to move forward into her/his research, that's a plus to me. Please read the disclaimer.


# The guide

The problem I faced was pretty simple. I wanted to know how to train an artificial neural network in PyTorch and how to convert this network into a CoreML model usable in an iOS application. Simple right?

I present bellow **my** solution divided into several steps:

- [Step 1](step1.md): Train a model using PyTorch and save it
- [Step 2](step2.md): Load the model and test it (for verification purpose)
- [Step 3](step3.md): Convert the PyTorch model into a ONNX model
- [Step 4](step4.md): Convert the ONNX model into a CoreML model
- [Step 5](step5.md): Test the CoreML model


# Virtual environment

For this work, I've been using [Conda](https://docs.conda.io) through [Anaconda](https://www.anaconda.com/) for (1) creating a virtual environment and (2) installing most of the used Python packages. Please read the official documentations for more information.


# Disclaimer

As presented in the _motivation_ section, the target audience of this guide is me. I am of course happy if it helps other coders around the world. I do not certify the code presented is the best or even the correct way to using PyTorch, ONNX, coremltools, etc. The code is probably uncomplete and might even contain serious bugs. In addition, this code will probably ceased to work with the newer versions of the different libraries and the evolution of PyTorch itself. Use this code at your own risk.

The code presented here has tested in 2020 on a MacBook Pro using:

- pytorch 1.5.0
- torchvision 0.6.0
- onnx 1.7.0
- onnx-coreml 1.3
- coremltools 3.4
- numpy 1.18.1
- pillow 7.0.0

If you see something wrong, please let me know and I'll be happy to make modifications.


# Acknowledgment

I've been reading intensively [PyTorch tutorials](https://pytorch.org/tutorials/) to educate myself. Don't be surprised if you find some similarities in the code.
