from ultralytics import YOLO
import cv2

def extract_detections_pt(yolo_result): 
    dets = []
    result = yolo_result.cpu()
    names = result.names
    #print(names)
    for i in range(result.boxes.xyxy.shape[0]): 
        d = {'ltrb':[int(el) for el in list(result.boxes.xyxy[i].numpy())],
             'score': result.boxes.conf[i].item(),
             'label': names[int(result.boxes.cls[i].item())],
             "categ_numb": int(result.boxes.cls[i].item())}
        dets.append(d)
    return dets

def draw_bbox(image, detections):
    for d in detections:
        cv2.rectangle(image, tuple(d['ltrb'][:2]),tuple(d["ltrb"][2:]), (255, 0, 0), 2)
        #print(d["label"])
    return image

def process_box(detections):
    xyxy = [each_box['ltrb'] for each_box in detections]
    conf = [each_box['score'] for each_box in detections]
    label = [each_box['label'] for each_box in detections]
    categ_n = [each_box['categ_numb'] for each_box in detections]
    print("xyxy_first image",xyxy)
    print("conf",conf)
    print("category",label)
    print("category_numb", categ_n)
    return 

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model
    yolo_results = model([ "/data/projects/ultralytics/ultralytics/assets/zidane.jpg", "/data/projects/ultralytics/ultralytics/assets/bus.jpg"])
    for i in range(len(yolo_results)):
        extract_detections_pt(yolo_results[i])

