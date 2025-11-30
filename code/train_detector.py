"""
Minimal YOLOv8 training wrapper.
Prepare `data.yaml` in YOLO format before running (train/val paths + nc/classes).
"""
from ultralytics import YOLO

def train(data_yaml='data.yaml', base_model='yolov8n.pt', epochs=50, imgsz=640):
    model = YOLO(base_model)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=16, name='ball_train')
    print('Training finished. Check runs/train/ball_train/weights/')

if __name__ == '__main__':
    train()
