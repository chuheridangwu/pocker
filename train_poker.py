from ultralytics import YOLO
import torch

def train_poker():
    # 1. 检查 Mac 的 GPU (MPS) 是否可用
    if torch.backends.mps.is_available():
        device = "mps"
        print("检测到 M4 GPU 加速可用！")
    else:
        device = "cpu"
        print("未检测到 MPS，将使用 CPU 训练（速度较慢）")

    # 2. 加载模型
    model = YOLO("yolo11n.pt")

    # 3. 开始训练
    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=416,
        batch=64,            # 16GB 内存设为 64 没问题
        workers=4,           # 建议先设为 4，Mac 上过高的 workers 有时会导致内存压力溢出
        device=device,
        cache=True,          # 把图片加载到 16GB 内存中，这是 M4 最强的加速手段
        plots=True,
        augment=True,        # 开启增强
        save=True
    )

if __name__ == "__main__":
    train_poker()
