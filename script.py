import json
import os
from collections import defaultdict
import random
import shutil
import cv2
import yaml
import argparse
#make dir empty before adding new data
def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

#converting data from coco format into yolo and saving to folder labels
# target_labels = dict(name : yolo_id)

def read_coco_to_yolo(coco_json_file, target_labels={}, sanity=False):
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)
    images = {img['id']: img for img in coco_data['images']}
    
    if target_labels:
        target_coco_categories = {cat['id']: target_labels[cat['name']] for cat in coco_data['categories'] if cat['name'] in target_labels}
    else:
        target_coco_categories = {cat['id']: id for id, cat in enumerate(coco_data['categories'])}
        coco_categories = {cat['id']:cat['name'] for cat in coco_data['categories']}
        target_labels = {yolo_id: coco_categories[coco_id]  for coco_id, yolo_id in target_coco_categories.items()}
    #print(f'TARGET {target_labels}')
    yolo_dict = defaultdict(list)
    sanity_images = set()
    for ann in coco_data['annotations']:
        category_id = ann['category_id']
        class_id = target_coco_categories.get(category_id, None)
        if class_id is None and target_labels:
            continue
        image_id = ann['image_id']
        image_info = images[image_id]
        file_name = os.path.basename(image_info['file_name'])
        x, y, w, h = ann['bbox']
        x_center = float(x + w/2) / image_info["width"]
        y_center = float(y + h/2) / image_info["height"]
        width = float(w) / image_info["width"]
        height = float(h) / image_info["height"]       
        yolo_ann =f"{class_id} {x_center} {y_center} {width} {height}"
        yolo_dict[file_name].append(yolo_ann)
        sanity_images.add(file_name)
    if sanity:
        sanity_images = list(sanity_images)
        random.shuffle(sanity_images)
        sanity_images= sanity_images[:10]
        yolo_dict = {img: yolo_dict[img] for img in sanity_images}, target_labels
        return yolo_dict
    return yolo_dict, target_labels

def create_yolo_label_txts(yolo_dict, labels_dir):
    clear_directory(labels_dir)
    replace_ext = lambda fn, new_ext: '.'.join(fn.split('.')[:-1] + [new_ext])
    counter = 0
    for filename, annotations in yolo_dict.items():
        yolo_file_path = os.path.join(labels_dir, replace_ext(filename, 'txt'))
        with open(yolo_file_path, 'w') as fp:
            annotations = list(set(annotations))
            for el in annotations:
                fp.write(el + '\n')
            counter += 1
    return counter

#copy images based on dict 
def copy_images(image_names, src_folder, dst_folder):
    clear_directory(dst_folder)
    counter = 0
    for image_name in image_names:
        src_image_path = os.path.join(src_folder, image_name)
        dst_image_path = os.path.join(dst_folder, image_name)
        
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)
            counter += 1
        else:
            print(f"Warning: {src_image_path} does not exist and will be skipped.")
    return counter

def create_subset(target_labels, coco_src_dir, dst, sanity):
    coco_ann_path = os.path.join(coco_src_dir, 'coco.json')
    coco_img_path = os.path.join(coco_src_dir, 'data')
    assert(os.path.isfile(coco_ann_path)), f"Cannot find coco.json file within {coco_src_dir}"
    assert(os.path.isdir(coco_img_path)),  f"Cannot find data file within {coco_src_dir}"
    yolo_dict, target_labels = read_coco_to_yolo(coco_ann_path, target_labels, sanity)
    num = create_yolo_label_txts(yolo_dict, os.path.join(dst, 'labels'))
    print(f"{num} label.txt files have been created within {dst}")
    num = copy_images(yolo_dict.keys(), coco_img_path, os.path.join(dst, 'images'))
    print(f"{num} images have been copied to {dst}")
    return dst, target_labels

def create_yaml_file(target_labels, outputs, output_root):
    data = {
        'paths':outputs,
        'nc': len(target_labels),
        'names': target_labels}
    output_file = os.path.join(output_root, 'yaml_file')
    with open(output_file, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    print(f'YAML file has been created at {output_file}')

def main(args):
    clear_directory(args.output_root)
    target_labels = {name:id for id,name in enumerate(args.classes)}
    coco_dirs = {'train': args.train, 'val': args.val, 'test': args.test}
    outputs = {}
    classnames = {}
    for subset in ['train', 'val', 'test']:
        outputs[subset], classnames[subset] = create_subset(target_labels, coco_dirs[subset], os.path.join(args.output_root, subset), args.sanity)
        #print(f'outputs:{outputs}')
        #print(f'classnames:{classnames}')
        if args.sanity:
            outputs['val'] = outputs['train']
            outputs['test'] = outputs['train']
            break
    if not args.sanity:
        if classnames['train'] != classnames['val'] or classnames['train'] != classnames['test'] or classnames['val'] != classnames['test']:
            print("WARNING: subsets of coco have diffent classnames:")
            for k,v in classnames: print(k, v)

    # write yaml file using outputs and target_labels
    yaml_path = create_yaml_file(classnames['train'], outputs, args.output_root)
    print(yaml_path)
    return outputs
def parse_args():
    parser = argparse.ArgumentParser("Make up YOLO dataset for train/val/test using FLIR dataset")
    parser.add_argument('--train', type=str, nargs='?', default=[], help="COCO train data folder(-s) (contains data subfolder and coco.json)")
    parser.add_argument('--val', type=str, nargs='?', default=[], help="COCO val data folder(-s) (contains data subfolder and coco.json)")
    parser.add_argument('--test', type=str, nargs='?', default=[], help="COCO test data folder(-s) (contains data subfolder and coco.json)")
    parser.add_argument('-s', '--sanity', type=str, nargs='?', default=[], help="Sanity check folder")
    parser.add_argument('-c', '--classes', type=str, nargs='+', default=[], help="Classes used")
    parser.add_argument('-o', '--output-root', type=str, required=True, help="Path where to put configured yolo v8 dataset")
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())