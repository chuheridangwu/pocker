
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os

def compare_conf_thresholds(image_path, model_path, conf_list=[0.1, 0.2, 0.3, 0.5]):
    # 1. 加载模型
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    
    model = YOLO(model_path)
    
    # 检查并使用 M4 的 GPU 加速
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"正在使用设备: {device} | 正在分析: {image_path}")

    # 读取原始图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return

    results_images = []

    for conf_val in conf_list:
        # 2. 推理预测
        # 对于扑克牌重叠严重的 111.jpg，建议用 imgsz=640 推理，效果往往比 416 好
        results = model.predict(
            source=img,
            conf=conf_val,
            device=device,
            imgsz=(384, 640),  # 传入符合 16:9 比例的尺寸（必须是 32 的倍数）            save=False,
            rect=True,         # 开启矩形推理，防止图像拉伸
            verbose=False # 减少控制台刷屏
        )
        
        # 3. 绘制识别结果到图片上
        # plot() 会返回带有识别框的 BGR 图像数组
        res_img = results[0].plot(line_width=2, font_size=1.0)
        
        # 在图片左上角用红色标注当前使用的 conf 值
        cv2.putText(res_img, f"Conf: {conf_val}", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
        
        results_images.append(res_img)
        print(f"阈值 {conf_val} 处理完成，检测到 {len(results[0].boxes)} 个目标。")

    # 4. 拼接图片 (2x2 网格)
    # 将 4 张图水平+垂直拼接
    top_row = np.hstack((results_images[0], results_images[1]))
    bottom_row = np.hstack((results_images[2], results_images[3]))
    combined = np.vstack((top_row, bottom_row))
    
    # 5. 保存并显示
    output_name = "comparison_result.jpg"
    cv2.imwrite(output_name, combined)
    print(f"\n对比图已保存至: {output_name}")

    # 缩放一下，防止图片太大超出 Mac 屏幕显示范围
    screen_h, screen_w = 720, 1280
    final_view = cv2.resize(combined, (screen_w, int(screen_w * combined.shape[0] / combined.shape[1])))
    
    cv2.imshow("M4 Poker - Conf Comparison (0.1, 0.2, 0.3, 0.5)", final_view)
    print("\n提示：按键盘任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- 请根据你的实际情况修改下面两个路径 ---
    target_image = "111.png"  # 你那张识别不全的游戏截图
    my_model = "runs/detect/train/weights/best.pt" # 你的模型权重路径
    
    compare_conf_thresholds(target_image, my_model)
