from ultralytics import YOLO
import cv2

# 加载你训练好的模型
model = YOLO("runs/detect/train/weights/best.pt")

# 指定视频文件路径
video_path = "111.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 在每一帧上运行推理
        # imgsz=416 保持和你训练时一致，device='mps' 利用 M4 GPU
        results = model.predict(frame, imgsz=640, conf=0.5, device='mps')

        # 绘制识别框并显示
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO 扑克牌识别", annotated_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
