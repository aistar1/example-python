# sample-python

## coco_filter_obj.py
功能: 從 coco data 中挑選自己想要的類別並轉換成特定格式去儲存物件的資訊(box, class)

## random_pickup.py
功能: 隨機挑選圖片

## read_onnx.py
功能: 讀取onnx格式模型

## test_dataloader.py
功能: 讀取dataset with pytorch
```text
dogDataset/
├── Images
│   ├── n02085620-Chihuahua
│   ├── ...
└── lists
    ├── file_list.mat
    ├── test_list.mat
    └── train_list.mat
```
reference:  
http://vision.stanford.edu/aditya86/ImageNetDogs/

## mosaic.py
功能: 將四張圖片合成一張
![image](https://user-images.githubusercontent.com/123159112/222131459-a8f366cb-6b3c-4b07-8e64-8bbeae47c73d.png)
