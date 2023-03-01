import random

import cv2
import os
import glob
import numpy as np
import json
import time

from PIL import Image

OUTPUT_SIZE = (640, 640)  # Height, Width
SCALE_RANGE = (0.3, 0.7)
FILTER_TINY_SCALE = -1#1 / 50  # if height or width lower than this scale, drop it.

ANNO_DIR = 'test/Annotations/'
IMG_DIR = 'test/Images/'



def main():
    img_paths, annos = get_dataset(ANNO_DIR, IMG_DIR)


    detections_json = []
    ob_num = 0
    image_id = 0
    images_dict = []
    annotations_json = []

    image_id_list = [i for i in range(len(annos))]
    result = len(image_id_list) % (2 * 2)
    if result != 0:
        #image_id_list.extend(image_id_list[: (2 * 2) - result])
        #img_paths.extend(img_paths[: (2 * 2) - result])
        #annos.extend(annos[: (2 * 2) - result])
        image_id_list = [i for i in range(len(annos) - result)]
        number = len(image_id_list) // (2 * 2)
    else:
        number = len(image_id_list) // (2 * 2)
    random.shuffle(image_id_list)
    for num in range(number):
        idxs = image_id_list[4*num : 4*num+4]
        mos_img,anno_json = update_image_and_anno(img_paths, annos,
                                                 idxs,
                                                 OUTPUT_SIZE, SCALE_RANGE,
                                                 filter_scale=FILTER_TINY_SCALE)

        img_name = time.strftime('face_%Y%m%d%H%M%S_') + str(num) + '.jpg'
        image_id = image_id + 1
        images_dict.append({
                    'id': image_id,
                    'file_name': img_name,
                    'width': mos_img.shape[1],
                    'height': mos_img.shape[0]})


        for i, anno in enumerate(anno_json):
            start_point = (int(anno[1]), int(anno[2]))
            end_point = (int(anno[3]), int(anno[4]))
            
            keypoint_list = []
            for num in range(5):
                keypoint_list.extend(anno[5][num])
                
            ob_num += 1
            annotations_json.append({
                            "keypoints": keypoint_list,
                            'bbox': [int(anno[1]), int(anno[2]), int(anno[3]) - int(anno[1]), int(anno[4]) - int(anno[2])],
                            'image_id': image_id, #!!!!! image nane
                            'category_id': anno[0],
                            'id': ob_num
                        })
            if True:
                cv2.rectangle(mos_img, start_point, end_point, (0, 255, 0), 3, cv2.LINE_AA)
                for l in range(5):
                    color = (255,0,0)
                    cv2.circle(mos_img, (int(anno[5][l][0]), int(anno[5][l][1])), 5, color, 2)
            
        cv2.imwrite(f'img/{img_name}', mos_img)
        

    detections_json = {'annotations': annotations_json}
    detections_json['images'] =  images_dict
    detections_json['categories'] = [{
                                            "supercategory": "face",
                                            "id": 1,
                                            "name": "face"
                                            }
                                            ]
    json.dump(detections_json, open("detections.json", 'w'), indent=4)



def load_image(path, img_size):

    small_img_h_target = img_size[0]
    small_img_w_target = img_size[1]
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    if h0 > w0:
        r = small_img_h_target / max(h0, w0)  # resize image to img_size
        #if r != 1:  # always resize down, only resize up if training with augmentation
        if small_img_w_target < r * w0:
            r = small_img_w_target / w0  # resize image to img_size
        interp = cv2.INTER_AREA 
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    else:
        r = small_img_w_target / max(h0, w0)  # resize image to img_size
        if small_img_h_target < r * h0:
            r = small_img_h_target / h0  # resize image to img_size
        #if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[1] = w * (x[1]) + padw  # top left x
    y[2] = h * (x[2]) + padh  # top left y
    y[3] = w * (x[3]) + padw  # bottom right x
    y[4] = h * (x[4]) + padh  # bottom right y
    for idx, point in enumerate(x[5]):
        y[5][idx] = [w * point[0]  + padw , h * point[1] + padh]
    return y

def update_image_and_anno(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0.):

    img_resize = output_size #h,w
    mosaic_border = [-img_resize[0] // 2, -img_resize[1] // 2]

    s = img_resize
    yc, xc = [int(random.uniform(-x, 2 * s[0] + x)) for x in mosaic_border]  # mosaic center x, y
    yc = img_resize[0]*2 // 2
    xc = img_resize[1]*2 // 2
    labels4 = []

    for i, idx in enumerate(idxs):
        path = all_img_list[idx]
        img_annos = all_annos[idx]

        # Load image
        img, _, (h, w) = load_image(path, img_resize)
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s[0] * 2, s[1] * 2, 3), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s[1] * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s[0] * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s[1] * 2), min(s[0] * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        for bbox in img_annos: # In one image all obj
            labels = xywhn2xyxy(bbox, w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels4.append(labels)


    return  img4, labels4


def get_dataset(anno_dir, img_dir):

    img_paths = []
    annos = []


    for anno_file in glob.glob(os.path.join(anno_dir, '*.json')):

        with open(anno_file, 'r') as f:
            data = json.load(f)

            for _, images_json in enumerate(data['images']):
                img_path = os.path.join(img_dir, images_json['file_name'])
                img = cv2.imread(img_path)
                img_height, img_width, _ = img.shape
                del img

                boxes = []
                for anno_json in data['annotations']:
                    keypoint_list = []
                    if images_json['id'] == anno_json['image_id']:
                        obj_bbox = anno_json['bbox']
                        xmin = max(obj_bbox[0], 0) / img_width
                        ymin = max(obj_bbox[1], 0) / img_height
                        xmax = min(obj_bbox[0] + obj_bbox[2], img_width) / img_width
                        ymax = min(obj_bbox[1] + obj_bbox[3], img_height) / img_height
                        keypoints = anno_json['keypoints']
                        for point in range(5):
                            keypoint_list.append([keypoints[point*2] / img_width, keypoints[point*2+1] / img_height])
                        class_id = anno_json['category_id']
                        boxes.append([class_id, xmin, ymin, xmax, ymax, keypoint_list])
                    if not boxes:
                        continue
                boxes = np.array(boxes,dtype=object)

                img_paths.append(img_path)
                annos.append(boxes)
    return img_paths, annos


if __name__ == '__main__':
    main()
