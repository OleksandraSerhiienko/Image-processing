from ultralytics import YOLO
import time
import os
import cv2
import argparse

def yolo_to_onnx(image, onnx_model):
    start_time=time.time()
    results = onnx_model(image)
    inf_time = time.time() - start_time 
    return inf_time


def main(args):
    yolo_model_path = args.yolo_model_path
    images_folder = args.images_folder
    img_ext = args.img_ext

    model = YOLO(yolo_model_path)
    onnx_model_path = model.export(format='onnx')
    onnx_model = YOLO(onnx_model_path, task='detect')

    total_inference_time = 0
    images_filepaths = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.endswith(img_ext)]
    num_images = len(images_filepaths)

    for i, image_filepath in enumerate(images_filepaths):
        print(f'I:{i}')
        if not os.path.isfile(image_filepath):
            print(f"File {image_filepath} does not exist")
            continue
        
        image = cv2.imread(image_filepath)
        if image is None:
            print(f"Cannot read image at {image_filepath}")
            continue
        inf_time = yolo_to_onnx(image, onnx_model)
        total_inference_time += inf_time
    if num_images > 0:
        print(f'Aver inference time: {round(1000. * total_inference_time/num_images, 2)} ms')

def parse_args():
    parser = argparse.ArgumentParser("Converting YOLO model into ONNX format")
    parser.add_argument('-m','--yolo-model-path', type=str, required=True, help="Path to yolo_model")
    parser.add_argument('-img', '--images-folder', type=str, required=True, help="Path to folder with images")
    parser.add_argument('-ext', '--img-ext', type = str ,default='.jpg', help='Image extension')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)



