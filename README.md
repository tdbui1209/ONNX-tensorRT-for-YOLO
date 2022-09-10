Tested with: Python 3.7.9, Pytorch 1.12.1+cu113, cuDNN 8.4, tensorRT 8.4

# Prepare
```
git clone https://github.com/WongKinYiu/yolov7.git/
git clone https://github.com/Linaom1214/tensorrt-python.git/
```

* download CUDA 11.3; cuDNN 8.4; tensorRT 8.4

* add to PATH CUDA\bin; CUDA11.3\libnvvp; cuDNN\bin; tensorRT\lib

* download zlib and copy zlibwapi.dll to Window\System32
(https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows)

* Install Microsoft C++ Build Tools
(https://visualstudio.microsoft.com/visual-cpp-build-tools/)

* Install pycuda
```
pip install pycuda
```

* Install pytorch
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

* Install remain dependencies
```
pip install -r requirements.txt
```

# Export

## YOLOv7

### FP16
```
python yolov7\export.py --weights path\to\pretrain\model\.pt --device 0 --grid --simplify --fp16
python tensorrt-python\export.py -o path\to\exported\.onnx -e path\to\exported\.trt -p fp16
```
### FP32
```
python yolov7\export.py --weights path\to\pretrain\model\.pt --device 0 --grid --simplify
python tensorrt-python\export.py -o path\to\exported\.onnx -e path\to\exported\.trt -p fp32
```

# Inference

## YOLOv5
|            | Time (ms) |
| -----------|------|
| onnx_FP32 | 9 |
| onnx_FP16 | 8 |
| trt_FP32 | 6 |
| trt_FP16 | 4 |

## YOLOv7
|            | Time (ms) |
| -----------|------|
| onnx_FP32 | 58 |
| onnx_FP16 | 58 |
| trt_FP32 | 42 |
| trt_FP16 | 24 |
