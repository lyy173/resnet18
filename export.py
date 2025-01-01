from resnet18 import resnet18
import torch

model = resnet18()
model.load_state_dict(torch.load('weights/best.pth'))

input_data = torch.zeros((1,3,224,224))
torch.onnx.export(model,input_data,"catdog.onnx")