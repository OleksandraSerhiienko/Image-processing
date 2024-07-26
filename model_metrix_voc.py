import cv2
import os
from random import shuffle
from ultralytics import YOLO
from yolo8_model_predict import extract_detections_pt
from utils_changed import crop_box, draw_objects, calculate_iou
from annotation_parsers import parse_voc_folder
import argparse
def draw_and_save_coco(coco_annotations, result_folder):
    os.makedirs(result_folder, exist_ok=True)
    for image_filepath, objects in coco_annotations.items():
        image = cv2.imread(image_filepath)
        if image is None:
            #print(f"Cannot read image. Check path {image_filepath}")
            continue
        image = draw_objects(image, objects)
        result_filepath = os.path.join(result_folder, os.path.split(image_filepath)[-1])
        cv2.imwrite(result_filepath, image)

def run_image_with_yolo(image, model):
    detections = []
    predicted_data = model(image)
    for pred in predicted_data:
        dets = extract_detections_pt(pred)
        for det in dets:
            label = det['label']
            score = det['score']
            box = crop_box(det['ltrb'], image.shape[:2])
            detections.append({'label': label, 'ltrb': box, 'score': score})
    return detections


def process_images(image_filepaths, model, annotations={}, result_folder=None):
    if result_folder: os.makedirs(result_folder, exist_ok=True)
    images_annotations_detections = {}
    for image_filepath in image_filepaths:
        if not os.path.isfile(image_filepath):
            continue
        # read image
        image = cv2.imread(image_filepath)
        if image is None:
            #print(f"Cannot read image. Check path {image_filepath}")
            continue
        annotated_objects = annotations.get(os.path.split(image_filepath)[-1], [])
        detections = run_image_with_yolo(image, model)
        images_annotations_detections[image_filepath] = {
            "annotated_objects": annotated_objects,
            "detections": detections
        }
        if result_folder:
            result_filepath = os.path.join(result_folder, os.path.split(image_filepath)[-1])
            image = draw_objects(image, annotated_objects)
            image = draw_objects(image, detections)
            cv2.imwrite(result_filepath, image)
            #for v in images_annotations_detections[image_filepath]["annotated_objects"]: print('annotated: ' + str(v))
            #for v in images_annotations_detections[image_filepath]["detections"]: print('detected: ' + str(v))
    return images_annotations_detections

def compute_ious(img_filepath, data):
    results = []
    ground_truth_data = data['annotated_objects']
    predicted_data = data['detections']
    for gt_obj in ground_truth_data:
            gt_label, gt_box = gt_obj['label'], gt_obj['ltrb']
            for detection in predicted_data:
                det_label, det_box, det_score = detection['label'], detection['ltrb'], detection['score']
                if gt_label == det_label:
                    iou = calculate_iou(gt_box, det_box)
                    results.append({
                        'img_filepath': img_filepath, 
                        'label': gt_label, 
                        'gt_box': gt_box,
                        'det_box': det_box,
                        'iou': iou})
    return results

def check_file_existence(filepath):
    if not os.path.isfile(filepath):
        #print(f'Warning file {filepath} does not exist')
        return False
    return True


def get_tp_fn_fp(ann_obj, detections, target_labels, iou_threshold=0.7, conf_threshold=0.7):
    def is_in(box, label, lst, iou_threshold, conf_threshold):
        for obj in lst:
            if not label == obj['label']: continue
            if obj.get('score', 1.0) <= conf_threshold: continue
            if calculate_iou(box, obj['ltrb']) < iou_threshold: continue
            return True
        return False
    
    tp = {}
    tp, fn = 0, 0
    for gt in ann_obj:
        if not gt['label'] in target_labels: continue
        if is_in(gt['ltrb'], gt['label'], detections, iou_threshold, conf_threshold): tp += 1
        else: fn += 1
    fp = 0
    for pred in detections:
        if not pred['label'] in target_labels: continue
        if is_in(pred['ltrb'], pred['label'], ann_obj, iou_threshold, conf_threshold): continue
        else: 
            #if pred['ltrb'][2] - pred['ltrb'][0] < 24 or pred['ltrb'][3] - pred['ltrb'][1] < 24: continue
            fp += 1   
    return tp, fn, fp

def evaluate(images_annotations_detections, target_labels, iou_threshold=0.7, conf_threshold=0.7):
    tps, fns, fps = 0, 0, 0
    for _, info in images_annotations_detections.items():
        annotated_obj = info["annotated_objects"]
        detections = info['detections']
        tp, fn, fp = get_tp_fn_fp(annotated_obj, detections, target_labels, iou_threshold, conf_threshold)
        tps += tp
        fns += fn
        fps += fp
    # pre, rec, f1s
    precision = tps/float(tps+fps) if tps+fps>0 else 0.0
    recall = tps/float(tps+fns) if tps+fns>0 else 0.0
    f1s = (2*precision*recall)/(precision+recall) if precision+recall>0 else 0.0
    return tps, fns, fps, precision, recall, f1s


def main(args):
    model_path = args.model_path
    annotation_filepath = args.labels
    images_folder = args.images
    vis_folder = args.save_path
    img_ext = args.img_ext
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    annotations={}
    if os.path.isdir(annotation_filepath):
        annotations = parse_voc_folder(annotation_filepath)
    else:
        print('No annotations by this path')
    model = YOLO(model_path)
    images_filepaths = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.endswith(img_ext)]
    images_annotations_detections = process_images(images_filepaths, model, annotations, vis_folder)
    tps, fns, fps, precision, recall, f1s = evaluate(images_annotations_detections, args.target_labels, iou_threshold=args.iou, conf_threshold=args.conf)
    print(f'TPS:{tps}, FNS:{fns}, FPS:{fps}, Precision:{precision}, recall:{recall}, f1score:{f1s}')
    print(f'Visualized images saved to {vis_folder}')
def parse_args():
    parser = argparse.ArgumentParser("YOLO model validation with metrix calculation")
    parser.add_argument('-ann','--labels', type=str, required=True, help="Path to folder with labels")
    parser.add_argument('-img', '--images', type=str, required=True, help="Path to folder with images")
    parser.add_argument('-ext', '--img-ext', type = str ,default='.jpg', help='Image extension')
    parser.add_argument('-e', '--epochs', type=int, default=25, help="Number of epochs for validation")
    parser.add_argument('-b', '--batch', type=float, default=16, help="Batch size for validation")
    parser.add_argument('-c', '--conf', type=float, default=0.001, help="Confidence threshold for validation")
    parser.add_argument('-iou', '--iou', type=float, default=0.5, help="IOU for validation")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for validation")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to run the validation on")
    parser.add_argument('-m', '--model_path', type=str,required=True, default='yolov8n.pt', help="Path to the YOLO model")
    parser.add_argument('-s', '--save_path', type=str,required=True, default='./results', help="Path to save the results")
    parser.add_argument('-tl', '--target-labels', nargs='+', default=[],help = 'Target labels' )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
 

 



