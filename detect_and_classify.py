import os
import cv2
import torch
from pathlib import Path
from race_classifier import predict_race

YOLO_PATH = os.path.join(os.getcwd(), r"D:\python\study\YOLOv5\YOLO-MFR\yolov5-master") #  改成你yolov5的路径
MODEL_PATH = r"best.pt"

# 加载 YOLOv5 模型
model = torch.hub.load(YOLO_PATH, 'custom', path=MODEL_PATH, source='local')
model.conf = 0.3
model.iou = 0.45
model.classes = None

COLOR_MAP = {
    "with_mask": (0, 255, 0),
    "no_mask": (0, 0, 255)
}

def process_frame(frame):
    results = model(frame)

    for i, (*xyxy, conf, cls) in enumerate(results.xyxy[0]):
        x1, y1, x2, y2 = map(int, xyxy)
        class_id = int(cls.item())
        label = model.names[class_id]

        face = frame[y1:y2, x1:x2]
        face_img_path = f"temp_face_{i}.jpg"
        cv2.imwrite(face_img_path, face)

        race, race_conf = predict_race(face_img_path)

        text = f"ID:{i} {label} {race} ({race_conf*100:.1f}%)"
        color = COLOR_MAP.get(label, (255, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        os.remove(face_img_path)

    return frame

def detect_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按下 q 键退出摄像头检测")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)
        cv2.imshow("Camera Detection", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_from_image(img_path):
    if not os.path.isfile(img_path):
        print("图片路径无效")
        return

    image = cv2.imread(img_path)
    processed = process_frame(image)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = r"D:\python\study\YOLOv5\YOLO-MFR\results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{base_name}_result.jpg")
    cv2.imwrite(save_path, processed)
    print(f"已保存检测结果至 {save_path}")

def detect_from_video(video_path):
    if not os.path.isfile(video_path):
        print("视频路径无效")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # 设置一个默认值

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("无法打开视频")
        return

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = r"D:\python\study\YOLOv5\YOLO-MFR\results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{base_name}_result.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps,
                          (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = process_frame(frame)
        out.write(processed)

    cap.release()
    out.release()
    print(f"已保存处理后的视频为{save_path} ")

def menu():
    while True:
        print("\n请选择模式：")
        print("0：退出")
        print("1：摄像头检测")
        print("2：图片检测")
        print("3：视频检测")

        choice = input("输入你的选择（0/1/2/3）：").strip()

        if choice == "0":
            print("已退出程序。")
            break
        elif choice == "1":
            detect_from_camera()
        elif choice == "2":
            path = input("请输入图片路径：").strip('" ')
            detect_from_image(path)
        elif choice == "3":
            path = input("请输入视频路径：").strip('" ')
            detect_from_video(path)
        else:
            print("无效选择，请重新输入。")

if __name__ == "__main__":
    menu()
