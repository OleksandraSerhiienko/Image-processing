import argparse
from ultralytics import YOLO
import os

def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    model = YOLO(args.model_path)
    results = model.train(data=args.yaml_path, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device, lr0=args.lr0,lrf=args.lrf, project= args.save_path, save_period = args.save_period)
    print(f"Training completed. Results saved to: {args.save_path}")

def parse_args():
    parser = argparse.ArgumentParser("YOLO model training")
    parser.add_argument('-y','--yaml_path', type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('-e', '--epochs', type=int, default=25, help="Number of epochs for training")
    parser.add_argument('-b', '--batch', type=float, default=0.6, help="Batch size for training")
    parser.add_argument('-sp', '--save-period', type=int, default=10, help="Save period for training")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the training on")
    parser.add_argument('--lr0', type=float, default=0.01, help="Initial learning rate")
    parser.add_argument('--lrf', type=float, default=0.01, help="Final learning rate")
    parser.add_argument('-m', '--model_path', type=str, default='yolov8n.pt', help="Path to the YOLO model")
    parser.add_argument('-s', '--save_path', type=str, default='./results', help="Path to save the results")
    return parser.parse_args()
 


if __name__ == '__main__':
    args = parse_args()
    main(args)