
from ultralytics import YOLO

# 加载你训练好的模型
model = YOLO("runs/detect/train/weights/best.pt")

# 打开摄像头进行实时预测
model.predict(source="0", show=True, device="mps")
