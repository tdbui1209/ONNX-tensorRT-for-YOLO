from refactor import Inference_onnx, Inference_trt
from utils import preprocesing, postprocessing
import cv2
import random
import os
import time


class_ = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

colors = {name:[random.randint(0, 255) for _ in range(3)] for name in class_}
infe = Inference_trt('yolov7.trt')

t1 = time.time()
for i in os.listdir('images'):
    img0 = cv2.imread(os.path.join('images', i))
    img, ratio, dwdh = preprocesing(img0)
    output = infe.run(img)
    # postprocessing(img0, ratio, dwdh, output)#, show_img=True, class_=class_, colors=colors)
print(time.time() - t1)