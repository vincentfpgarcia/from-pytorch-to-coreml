# Display CoreML model specifications

Coremltools is a set of tools developed by Apple. Among other things, it can display the specifications of a given CoreML model and use this model to perform the inference directly in Python. In other words, it allows to verify and test the model before deploying it on mobile.

In the following code, I simply load the CoreML model and display its specifications. The original code presented here can be found in the file [step4_part2.py](step4_part2.py)


```python
import coremltools

# Load the CoreML model
model =  coremltools.models.MLModel('my_network.mlmodel')

# Display its specifications
print(model.visualize_spec)
```

By executing these 2 lines of code, the print function displays the following information:

```
<bound method MLModel.visualize_spec of input {
  name: "my_input"
  shortDescription: "MultiArray of shape (1, 1, 3, 32, 32). The first and second dimensions correspond to sequence and batch size, respectively"
  type {
    multiArrayType {
      shape: 3
      shape: 32
      shape: 32
      dataType: FLOAT32
    }
  }
}
output {
  name: "my_output"
  shortDescription: "MultiArray of shape (1, 1, 10, 1, 1). The first and second dimensions correspond to sequence and batch size, respectively"
  type {
    multiArrayType {
      dataType: FLOAT32
    }
  }
}
metadata {
  userDefined {
    key: "coremltoolsVersion"
    value: "3.4"
  }
}
>
```

One can see the model specifications with the input / output names, sizes and types. This is very useful to use a given model without knowing anything about it. One important information here is that the model input and output types are `multiArrayType`. In iOS, this corresponds to the [MLMultiArray](https://developer.apple.com/documentation/coreml/mlmultiarray) type. But in Python, since we don't have access to this type, we need to manipulate Numpy array, which is pretty convenient.