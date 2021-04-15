# Display CoreML model specifications

As in [step 4](step4_part2.md), I use coremltools to display the CoreML model specifications.

The original code presented here can be found in the file [step6_part2.py](step6_part2.py)


```python
import coremltools

# Load the CoreML model
model =  coremltools.models.MLModel('my_network_image_ct4.mlmodel')

# Display its specifications
print(model)
```

The specifications are:

```
input {
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
      dataType: FLOAT32
    }
  }
}
metadata {
  userDefined {
    key: "com.github.apple.coremltools.source"
    value: "torch==1.8.0"
  }
  userDefined {
    key: "com.github.apple.coremltools.version"
    value: "4.1"
  }
}
```
