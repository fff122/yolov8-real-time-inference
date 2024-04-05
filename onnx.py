from ultralytics import YOLO

#加载模型填你的模型路径
model = YOLO(model=r'your_model.pt')

#导出onnx格式
model.export(format= "onnx")