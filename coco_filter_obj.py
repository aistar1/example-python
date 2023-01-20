from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
import json
from PIL import Image, ImageDraw

# the path you want to save your results for coco to voc
savepath = "coco2017/"
img_dir = savepath + 'images/'
anno_dir = savepath + 'Annotations/'
# datasets_list=['train2014', 'val2014']
#datasets_list = ['train2017','val2017']
datasets_list = ['train2017']

#classes_names = ['car', 'bicycle', 'person', 'motorcycle', 'bus', 'truck']
classes_names = ['person', 'car', 'bus', 'truck', 'bicycle',  'motorcycle']
# Store annotations and train2014/val2014/... in this folder
dataDir = '.'



# if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


mkr(img_dir)
mkr(anno_dir)

def bbox_(x, y, w, h, label='person'):
    '''
        x, y: top-left position
        w: width
        h: height
        label: default "2", if person bbox set "3"
    '''
    ret = {
            "label": label,
            "points": [
            [
                x,
                y
            ],
            [
                x + w,
                y + h
            ]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}  
        }
    return ret

def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def write_xml(anno_path, head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr % (obj[0], obj[1], obj[2], obj[3], obj[4]))
    f.write(tail)


def save_annotations_and_imgs(coco, dataset, filename, objs):
    # eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path = anno_dir + 'coco2017_'+filename[:-3] + 'json'
    img_path = dataDir + '/'+dataset + '/' + filename
    dst_imgpath = img_dir + 'coco2017_' + filename
    img = cv2.imread(img_path)
    if (img.shape[2] == 1):
        print(filename + " not a RGB image")
        return
    shutil.copy(img_path, dst_imgpath)

    
    result_data = {
            "version": "4.2.7",
            "flags": {},
            "shapes": objs,
            "imagePath": 'coco2017_' + filename,
            "imageData": None,
            "imageHeight": img.shape[0],
            "imageWidth": img.shape[1]
    }
    result_json = json.dumps(result_data, indent=10)
    with open(anno_path, 'w') as fp:
        fp.write(result_json)
    


def showimg(coco, dataset, img, classes, cls_id, show=True):
    global dataDir
    I = Image.open('%s/%s/%s' % (dataDir, dataset, img['file_name']))
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name = classes[ann['category_id']]
        if class_name in classes_names:
            #print(class_name)
            if 'iscrowd' in ann:
                if ann['iscrowd'] != 0:
                    continue
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                if class_name == 'person':
                    objs.append(bbox_(xmin, ymin, int(bbox[2]), int(bbox[3]), label = 'person'))
                elif class_name == 'car' or class_name == 'bus' or class_name == 'truck':
                    objs.append(bbox_(xmin, ymin, int(bbox[2]), int(bbox[3]), label = 'car'))
                elif class_name == 'bicycle' or class_name == 'motorcycle':
                    objs.append(bbox_(xmin, ymin, int(bbox[2]), int(bbox[3]), label = 'bicycle'))
                #draw = ImageDraw.Draw(I)
                #draw.rectangle([xmin, ymin, xmax, ymax])
            
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    return objs


for dataset in datasets_list:
    # ./COCO/annotations/instances_train2014.json
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)
    # COCO API for initializing annotated data
    coco = COCO(annFile)
    # show all classes in coco
    classes = id2name(coco)
    #print(classes)
    # [1, 2, 3, 4, 6, 8]
    classes_ids = coco.getCatIds(catNms=classes_names)
    #print(classes_ids)
    for cls in classes_names:
        # Get ID number of this class
        cls_id = coco.getCatIds(catNms=[cls])
        img_ids = coco.getImgIds(catIds=cls_id)
        #print(cls, len(img_ids))
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            #print("filename:",filename)
            #print("dataset:",dataset)
            objs = showimg(coco, dataset, img, classes, classes_ids, show=False)
            #print(objs)
            save_annotations_and_imgs(coco, dataset, filename, objs)
