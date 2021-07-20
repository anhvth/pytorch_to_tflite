# Install



`pip install pytorch-to-tflite`

or

`pip install git+https://github.com/anhvth/pytorch_to_tflite/`

# How to use

## Pytorch to Onnx

```python
from pytorch_to_tflite.pytorch_to_tflite import *
import torch
import yaml
import os
import mmcv
from nanodet.model.arch import build_model

PATH_TO_CONFIG = '/gitprojects/nano-det-parkingline/config/nanodet-g.yml'
cfg = yaml.safe_load(open(PATH_TO_CONFIG))
cfg = mmcv.Config(cfg)
model = build_model(cfg.model)

img = torch.randn(1,3,416,416)
out = model(img)

!mkdir -p cache/
onnx_out_path = 'cache/out.onnx'
torch.onnx.export(model, img, onnx_out_path)
```

    Finish initialize Lite GFL Head.


    /root/miniconda3/envs/pytorch-to-tflite/lib/python3.9/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    /root/miniconda3/envs/pytorch-to-tflite/lib/python3.9/site-packages/torch/nn/functional.py:3609: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
      warnings.warn(
    /root/miniconda3/envs/pytorch-to-tflite/lib/python3.9/site-packages/torch/nn/functional.py:3657: UserWarning: The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior, please set recompute_scale_factor=True. See the documentation of nn.Upsample for details. 
      warnings.warn(
    /root/miniconda3/envs/pytorch-to-tflite/lib/python3.9/site-packages/torch/onnx/symbolic_helper.py:374: UserWarning: You are trying to export the model with onnx:Upsample for ONNX opset version 9. This operator might cause results to not match the expected results by PyTorch.
    ONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11. Attributes to determine how to transform the input were added in onnx:Resize in opset 11 to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).
    We recommend using opset 11 and above for models using this operator. 
      warnings.warn("You are trying to export the model with " + onnx_op + " for ONNX opset version "


# ONNX to Tensorflow

```python
onnx_path = onnx_out_path
tf_path = onnx_path + '.tf'
onnx_to_tf(onnx_path=onnx_path, tf_path=tf_path)
assert os.path.exists(tf_path)
```

    WARNING:tensorflow:From /root/miniconda3/envs/pytorch-to-tflite/lib/python3.9/site-packages/tensorflow/python/ops/array_ops.py:5043: calling gather (from tensorflow.python.ops.array_ops) with validate_indices is deprecated and will be removed in a future version.
    Instructions for updating:
    The `validate_indices` argument has no effect. Indices are always validated on CPU and never validated on GPU.


    WARNING:absl:Function `__call__` contains input name(s) input.1 with unsupported characters which will be renamed to input_1 in the SavedModel.
    WARNING:absl:Found untraced functions such as gen_tensor_dict while saving (showing 1 of 1). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: cache/out.onnx.tf/assets


    INFO:tensorflow:Assets written to: cache/out.onnx.tf/assets


# Tensorflow to tflite

```python
tflite_path = tf_path+'.tflite'
tf_to_tf_lite(tf_path, tflite_path)
assert os.path.exists(tflite_path)
tflite_path
```




    'cache/out.onnx.tf.tflite'


