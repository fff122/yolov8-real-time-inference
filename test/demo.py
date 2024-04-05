import unittest
import cv2
from ultralytics import YOLO

class TestYOLOv8(unittest.TestCase):

    def test_yolov8n(self):
        # 加载模型
        model = YOLO(model="yolov8n.pt")  # 注意这里不需要完整的路径

        # 摄像头编号
        camera_no = 0

        # 打开摄像头
        cap = cv2.VideoCapture(camera_no)

        # 检查摄像头是否打开
        self.assertTrue(cap.isOpened())

        while cap.isOpened():
            # 获取图像
            ret, frame = cap.read()
            # 如果读取成功
            self.assertTrue(ret)
            # 正向推理
            results = model(frame)

            # 绘制结果
            annotated_frame = results[0].plot()

            # 显示图像
            cv2.imshow("YOLOv8", annotated_frame)

            # 按ESC退出
            if cv2.waitKey(1) == 27:
                break

        # 释放链接
        cap.release()
        # 销毁所有窗口
        cv2.destroyAllWindows()

if __name__ == '__main__':
    unittest.main()

