from ultralytics import YOLO

# 这里的路径现在是相对于你项目根目录的相对路径
model = YOLO("runs/detect/train2/weights/last.pt")

# 开启恢复模式
model.train(resume=True)
