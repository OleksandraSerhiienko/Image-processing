""""Parsers for VOC and COCO data annotations parsing"""

import os
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import json
from collections import defaultdict
def parse_voc(ann_path):
    image_name = os.path.split(ann_path)[1].replace('.xml', '.jpg')
    ann_dict = {image_name:[]}
    try:
        xml_data = open(ann_path, 'r').read()
        data = BeautifulSoup(xml_data, "xml")
        for obj in data.find_all('object'):
            classname = obj.find('name').text
            bbox = obj.find('bndbox')
            if bbox:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                object = dict(label=classname,ltrb=[xmin, ymin, xmax, ymax])
                ann_dict[image_name].append(object)
    except:
        print()
        return None
    return ann_dict

def parse_voc_folder(ann_folder):
    ann_dict = {}
    for ann_file in [el for el in os.listdir(ann_folder) if el.endswith('.xml')]:
        annotation = parse_voc(os.path.join(ann_folder, ann_file))
        if annotation:
            ann_dict.update(annotation)
    return ann_dict


def unit_test_voc():
    xml = "<annotation><filename>1522874691762_2.jpg</filename><folder>data</folder><segmented>0</segmented><size> <width>1920</width><height>1072</height><depth>3</depth></size><object><name>__background__</name><pose>Unspecified</pose><difficult>0</difficult><truncated>0</truncated><bndbox><xmin>752</xmin><ymin>136</ymin><xmax>814</xmax><ymax>226</ymax></bndbox></object></annotation>"
    root = ET.fromstring(xml)
    xml_file = ET.ElementTree(root)
    xml_path = '1522874691762_2.xml'
    xml_file.write(xml_path, encoding='utf-8')
    annotation = parse_voc(xml_path)
    #print(annotation)
    if annotation is None:
        print('Cannot parse xml')
        return False
    if '1522874691762_2.jpg' not in annotation: 
        print('No such image')
        return False
    for filename, obj in annotation.items():
        if not obj[0]['label'] == '__background__':
            print('No such category')
            return False
        if not obj[0]['ltrb'][0] == 752:
            print('No such xmin')
            return False
        if not obj[0]['ltrb'][1] == 136:
            print('No such ymin')
            return False
        if not obj[0]['ltrb'][2] == 814:
            print('No such xmax')
            return False
        if not obj[0]['ltrb'][3] == 226:
            print('No such ymax')
            return False
    os.remove(xml_path)
    return True



def parse_coco(ann_file):
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    yolo_dict = defaultdict(list)
    
    for ann in coco_data['annotations']:
        category_id = ann['category_id']
        class_name = categories.get(category_id, None)
        image_id = ann['image_id']
        image_info = images[image_id]
        file_name = os.path.basename(image_info['file_name'])
        
        xmin, ymin, w, h = ann['bbox']
        
        yolo_ann = dict(label=class_name, xmin=xmin, ymin=ymin, width=w, height=h)
        yolo_dict[file_name].append(yolo_ann)
    return yolo_dict

def unit_test_coco():
    coco_data = {
        'annotations': [
            {'area' : 576,
            "bbox": [237, 151,24,24],
            'category_id': 1, 
            'id': 1,
            'image_id': 1,
            'iscrowd' : 0}],
        'images': [{'id':1, 'width':640, 'height':480, 'file_name':'image1.jpg'}], 
        'categories':[{'id':1,'name':'person'}]}
    
    json_path = 'test_coco.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f)
    coco_dict = parse_coco(json_path)
    #print(coco_dict)
    if not coco_dict:
        print('Cannot parse JSON')
    for image_name, el in coco_dict.items():
        if not(image_name == 'image1.jpg'):
            print('No such image')
            return False
        if not el[0]['label'] =='person':
            print('No such class id')
            return False
        if not el[0]['xmin'] == 237:
            print('No such xmin value')
            return False
        if not el[0]['ymin'] == 151:
            print('No such ymin value')
            return False
        if not el[0]['width'] == 24:
            print('No such width value')
            return False
        if not el[0]['height'] == 24:
            print('No such height value')
            return False
    os.remove(json_path)
    return True


if __name__ == '__main__':
    assert(unit_test_voc()),"Unit test for VOC parsing has not been passed"
    print("All unit test have been successfully passed")
    assert(unit_test_coco()),"Unit test for COCO parsing has not been passed"
    print("All unit test have been successfully passed")