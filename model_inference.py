import torch
import cv2
import numpy as np
from utils import non_max_suppression

import time

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def preprocesing(img, img_size=(640, 640), fp16=False):
    # img = cv2.resize(img, img_size)
    img, ratio, dwdh = letterbox(img, auto=False)
    img = img[:, :, ::-1].transpose(2, 0, 1)

    if fp16:
        img = np.ascontiguousarray([img], dtype=np.float16)
    else:
        img = np.ascontiguousarray([img], dtype=np.float32)

    # normalize
    img /= 255.0
    return img, ratio, dwdh


def inference_onnx(img, onnx_path, v7=True, fp16=False):
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
    if not v7:
        # USE FP16?
        if fp16:
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


def postprocessing(img, ratio, dwdh, output, conf_threshold=0.25, iou_threshold=0.45, save_img=None, show_img=True, class_=None):
    pred = non_max_suppression(output, conf_threshold, iou_threshold)[0]
    for i, (x0, y0, x1, y1, score, cls_id) in enumerate(pred):
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        score = round(float(score), 3)
        cls_id = int(cls_id)

        cv2.rectangle(img, box[:2], box[2:], (0, 0, 255), 1)
        cv2.putText(img, class_[cls_id], (box[0] + box[2] - box[0] - 2, box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=1)
        cv2.putText(img, str(round(float(score), 2)), (box[0], box[1] - 2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        
    # for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(output):
        # box = np.array([x0,y0,x1,y1])
        # box -= np.array(dwdh*2)
        # box /= ratio
        # box = box.round().astype(np.int32).tolist()
        # cls_id = int(cls_id)
        
        # score = round(float(score),3)
        # cv2.rectangle(img, box[:2], box[2:], (0, 0, 255), 1)
        # cv2.putText(img, str(cls_id), (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=1)
        # cv2.putText(img, str(round(float(score), 2)), (box[0], box[1] - 2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    if save_img:
        cv2.imwrite(save_img, img)
    
    if show_img:
        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    class_ = ['car']
    img = cv2.imread('vid_5_31600.jpg')
    output, ratio, dwdh = inference_onnx(img, 'best_v7_16.onnx')
    postprocessing(img, ratio, dwdh, output, save_img='test.jpg', show_img=True, class_=class_)

    img = cv2.imread('vid_5_31600.jpg')
    output, ratio, dwdh = inference_trt(img, 'best_v7_16.trt')
    postprocessing(img, ratio, dwdh, output, save_img='test.jpg', show_img=True, class_=class_)
