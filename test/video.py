from ultralytics import YOLO
import cv2  # 确保已经正确安装了OpenCV库

# 加载模型
model = YOLO(model="yolov8n.pt")

# 视频文件路径这里填你的视频文件路径
video_path = "your_video.mp4"

# 打开视频文件
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # 读取视频帧
    ret, frame = cap.read()
    # 如果读取成功
    if ret:
        # 进行推理
        results = model(frame)

        # 绘制推理结果
        annotated_frame = results[0].plot()

        # 显示结果
        cv2.imshow("YOLOV8", annotated_frame)

        # 按ESC键退出
        if cv2.waitKey(1) == 27:
            break
    else:
        break

# 释放视频流和窗口资源
cap.release()
cv2.destroyAllWindows()
