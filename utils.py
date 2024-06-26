"""Utils for Dataset parsing, reading, writing.
Includes image/video read/write, draw ops."""

__author__ = "Aleksej Antonenko"

import os
import json
from collections import defaultdict
import cv2

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    #print(f'box1:{box1}')
    x2_min, y2_min, x2_max, y2_max = box2
    #print(f'box2:{box2}')
    #coordinates of intersection rectangle
    x1 = max(x1_min, x2_min)
    y1 = max(y1_min, y2_min)
    x2 = min(x1_max, x2_max)
    y2 = min(y1_max, y2_max)
    #print(x1,y1,x2,y2)
    intersect_width = max(0, x2 - x1)
    #print(f'intersect_width:{intersect_width}')
    intersect_height =  max(0, y2 - y1)
    #print(f'intersect_height:{intersect_height}')
    intersect_area = intersect_width*intersect_height
    #print(f'intersect_area:{intersect_area}')
    #compute the areas
    box1_area = (x1_max -x1_min)*(y1_max-y1_min)
    box2_area = (x2_max -x2_min)*(y2_max-y2_min)
    #union area
    union_area= box1_area + box2_area - intersect_area
    #print(f'union:{union_area}')
    if union_area <=0:
        return 0.0
    iou = intersect_area / float(union_area)
    #print(f'iou:{iou}')
    return iou

def crop_box(box, img_height_width):
    l,t,r,b = box
    l = max(0, l)
    t = max(0, t)
    r = min(img_height_width[1] - 1, r)
    b = min(img_height_width[0] - 1, b)
    return [l, t, r, b]

def parse_coco_dataset(images_folder, annotations_file):
    assert(os.path.isdir(images_folder)),f"Images folder does not exist. Check path: {images_folder}"
    assert(os.path.isfile(annotations_file)),f"Annotations file does not exist. Check path: {annotations_file}"

    coco_annotations = json.load(open(annotations_file))
    image_id_to_filename = {image['id'] : image['file_name'] for image in coco_annotations['images']}
    category_id_to_name = {category["id"] : category["name"] for category in coco_annotations["categories"]}
    annotations = defaultdict(list)
    for annotation in coco_annotations["annotations"]:
        image_filename = image_id_to_filename[annotation["image_id"]]
        image_filepath = os.path.join(images_folder, image_filename)
        assert(os.path.isfile(image_filepath)),f"Cannot find annotated image in provided folder. Check path: {image_filepath}"
        category_name = category_id_to_name[annotation["category_id"]]
        l, t, w, h = annotation["bbox"]
        r, b = l + w, t + h
        annotations[image_filepath].append({"label": category_name, "ltrb": [l, t, r, b]})
    return annotations

def draw_object(image, obj):
    l, t, r, b = crop_box(obj["ltrb"], image.shape[:2])
    cv2.rectangle(image, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
    text = obj["label"]
    color = (0, 255, 0)
    if "score" in obj:
        text += f" ({round(obj['score'], 2)})"
        color = (255, 0, 0)
    cv2.putText(image, text, (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def draw_objects(image, objects):
    for obj in objects:
        image = draw_object(image, obj)
    return image
