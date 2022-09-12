import cv2
import numpy as np
from utils import *

import time


def inference_onnx(img, onnx_path, v7=True, fp16=False):
    import torch
    import onnx
    import onnxruntime

    # CHECK MODEL ONNX
    onnx_model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s" % e)
    else:
        print("The model is valid!")

        # USE FP16?
        if fp16:
            img, ratio, dwdh = preprocesing(img, fp16=True)
        else:
            img, ratio, dwdh = preprocesing(img)
        # PREPARE
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(onnx_path, providers=providers)

        # WARMUP
        for _ in range(10):
            session.run(None, {'images':  img})[0]

        t1 = time.time()
        # INFERENCE
        src = session.run(None, {'images':  img})[0]
        print(round((time.time() - t1) * 1000), 'ms')

        src = torch.tensor(src)
        return src, ratio, dwdh

def inference_trt(img, engine_path, v7=True, fp16=False):
    import torch
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt

    # PREPARE
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        serialized_engine = f.read()

    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    for idx, binding in enumerate(engine):
        if engine.binding_is_input(binding):
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
            device_input = cuda.mem_alloc(input_size)
        else:
            output_shape = engine.get_binding_shape(binding)

    stream = cuda.Stream()

    # USE YOLOv7?
    if not v7 and fp16:
    # USE FP16?
        host_input, ratio, dwdh = preprocesing(img, fp16=True)
        host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float16)
    else:
        host_input, ratio, dwdh = preprocesing(img)
        host_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    device_output = cuda.mem_alloc(host_output.nbytes)

    # WARMUP
    for _ in range(10):
        cuda.memcpy_htod_async(device_input, host_input, stream)
        context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)

        cuda.memcpy_dtoh_async(host_output, device_output, stream)
        stream.synchronize()

    t1 = time.time()
    # INFERENCE
    cuda.memcpy_htod_async(device_input, host_input, stream)
    context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    print(round((time.time() - t1) * 1000), 'ms')

    host_output = torch.tensor(host_output.reshape(output_shape))
    return host_output, ratio, dwdh


if __name__ == '__main__':
    class_ = ['car']
    img = cv2.imread('vid_5_31600.jpg')
    output, ratio, dwdh = inference_onnx(img, 'best_v7_16.onnx')
    postprocessing(img, ratio, dwdh, output, save_img='test.jpg', show_img=True, class_=class_)

    img = cv2.imread('vid_5_31600.jpg')
    output, ratio, dwdh = inference_trt(img, 'best_v7_16.trt')
    postprocessing(img, ratio, dwdh, output, save_img='test.jpg', show_img=True, class_=class_)
