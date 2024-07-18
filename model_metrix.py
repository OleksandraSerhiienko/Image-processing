import json
import cv2
import os
from ultralytics import YOLO
from YOLOv8_predict import extract_detections_pt
from utils_changed import parse_coco_dataset, crop_box, draw_objects, calculate_iou
import shutil

folder_with_images_val = '/data/datasets/model_validation/split_train_val/val_dir/images'
folder_with_images_train = '/data/datasets/model_validation/split_train_val/train_dir/images'

coco_annotations_file = '/data/datasets/model_validation/coco/annotations/instances_default.json'
vis_images = 'vis_images'
yolo_model_name = "yolov8n"
#  functions
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
        # get annotated objects if annotations provided
        annotated_objects = annotations.get(image_filepath, [])
        # get detections
        detections = run_image_with_yolo(image, model)
        # draw objects and save an image if required
        if result_folder:
            result_filepath = os.path.join(result_folder, os.path.split(image_filepath)[-1])
            image = draw_objects(image, annotated_objects)
            image = draw_objects(image, detections)
            cv2.imwrite(result_filepath, image)
        #  keep results for next evaluation
        images_annotations_detections[image_filepath] = {
            "annotated_objects": annotated_objects,
            "detections": detections
        }
    return images_annotations_detections
coco_annotations = parse_coco_dataset(folder_with_images_val, coco_annotations_file)
model = YOLO(yolo_model_name)
images_annotations_detections = process_images(coco_annotations.keys(), model, coco_annotations, vis_images)

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
 
img_filepaths = list(coco_annotations.keys())
for img_filepath in img_filepaths:
    if check_file_existence(img_filepath):
        results = compute_ious(img_filepath, images_annotations_detections[img_filepath])
def get_tp_fn_fp(ann_obj, detections, iou_threshold=0.7, conf_threshold=0.8):
    def is_in(box, label, lst, iou_threshold, conf_threshold):
        for obj in lst:
            if not label == obj['label']: continue
            if obj.get('score', 1.0) <= conf_threshold: continue
            if calculate_iou(box, obj['ltrb']) < iou_threshold: continue
            return True
        return False
    
    tp, fn = 0, 0
    for gt in ann_obj:
        if is_in(gt['ltrb'], gt['label'], detections, iou_threshold, conf_threshold): tp += 1
        else: fn += 1
    fp = 0
    for pred in detections:
        if is_in(pred['ltrb'], pred['label'], ann_obj, iou_threshold, conf_threshold): continue
        else: fp += 1   
    return tp, fn, fp

def evaluate(images_annotations_detections, iou_threshold=0.7, conf_threshold=0.8):
    tps, fns, fps = 0, 0, 0
    for _, info in images_annotations_detections.items():
        annotated_obj = info["annotated_objects"]
        detections = info['detections']
        tp, fn, fp = get_tp_fn_fp(annotated_obj, detections, iou_threshold, conf_threshold)
        tps += tp
        fns += fn
        fps += fp
    # pre, rec, f1s
    precision = tps/float(tps+fps) if tps+fps>0 else 0.0
    recall = tps/float(tps+fns) if tps+fns>0 else 0.0
    f1s = (2*precision*recall)/(precision+recall) if precision+recall>0 else 0.0
    return tps, fns, fps, precision, recall, f1s

img_filepaths = list(coco_annotations.keys())
image_filepath  = img_filepaths[0]

tps, fns, fps, precision, recall, f1s = evaluate(images_annotations_detections, iou_threshold=0.7, conf_threshold=0.8)
print(f'Precision:{precision}, recal:{recall}, f1score:{f1s}')