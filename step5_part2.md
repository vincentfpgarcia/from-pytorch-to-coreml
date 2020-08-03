# Display CoreML model specifications

As in [step 4](step4_part2.md), I use coremltools to display the CoreML model specifications.

The original code presented here can be found in the file [step5_part2.py](step5_part2.py)


```python
import coremltools

# Load the CoreML model
model =  coremltools.models.MLModel('my_network_image.mlmodel')

# Display its specifications
print(model.visualize_spec)
```

The specifications are:

```
<bound method MLModel.visualize_spec of input {
  name: "my_input"
  type {
    imageType {
      width: 32
      height: 32
      colorSpace: RGB
    }
  }
}
output {
  name: "my_output"
  type {
    multiArrayType {
      shape: 1
      shape: 10
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

The input `my_input` is now of type `imageType` (compared to `multiArrayType` in step 4). In iOS, this corresponds to the `CVPixelBuffer` type, which is much more common that the previously used `MLMultiArray` type. Bonus: Using the Vision framework, the CoreML model can accept a `CGImage` as input, which is even more common and easily created from a `UIImage`!

In Python, the CoreML model accepts inputs of type PIL image, which is very convenient. We'll see why in the [next part](step5_part3.md).