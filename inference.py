import onnxruntime as ort
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

classdict = {0:"cat",1:"dog"}

session = ort.InferenceSession("./catdog.onnx",providers=["CUDAExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

testpath = './datasets/data/test'

def preprocess(image):
    image_o = image.copy()
    image_o = np.array(image_o.resize((224,224)))
    image_o = (image_o/255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    #增加一个batch维度，把颜色通道放到第二维
    image_o = image_o.transpose((2,0,1))[np.newaxis,:]
    return image_o.astype(np.float32)

while True:
    image_name = input("请输入图片名称：")
    image = Image.open(os.path.join(testpath,image_name+'.jpg'))
    image_o = preprocess(image)
    output = session.run(output_names=[output_name],input_feed={input_name:image_o})[0]
    pred = classdict[np.argmax(output)]
    print(pred)


