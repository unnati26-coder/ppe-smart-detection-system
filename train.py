from ultralytics import YOLO

def train_ppe_model():
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano model (fast), use yolov8m.pt for better accuracy

    # Train the model
    results = model.train(
        data='dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='ppe_detector',
        patience=10,           # early stopping
        device=0,              # GPU (use 'cpu' if no GPU)
        augment=True,          # data augmentation
        mosaic=1.0,
        degrees=10.0,
        flipud=0.5,
        fliplr=0.5,
        save=True,
        plots=True
    )
    
    print(f"Training complete! Best model saved to: runs/detect/ppe_detector/weights/best.pt")
    return results

if __name__ == '__main__':
    train_ppe_model()