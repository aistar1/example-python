import argparse
from random import shuffle
from scipy.io import loadmat
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

class customDataset(Dataset):
    def __init__(self, root, split, transform):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.transform = transform
        # Load image path and annotations
        mat = loadmat(f'{root}/lists/{split}_list.mat', squeeze_me=True)
        file_list = mat['file_list']
        self.imgs = [f'{root}/Images/{i}' for i in file_list]
        self.lbls = mat['labels']
        assert len(self.imgs) == len(self.lbls), 'mismatched length!'
        print(f'{split} data: {len(self.imgs)}')
        # Label from 0 to (len-1)
        self.lbls = self.lbls - 1

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------

        image_path = self.imgs[index]
        img = Image.open(image_path)
        label = int(self.lbls[index])
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.imgs)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dogDataset', help='data path')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=128, help='train image size (pixels)')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    return parser.parse_known_args()[0]

def main(opt):
    # Create train/valid transforms
    target_height = opt.imgsz * 2
    target_width = opt.imgsz
    # PIL -> RGB,  opencv -> BGR
    train_transform = T.Compose([
        T.Resize((target_height,target_width)),
        T.RandomCrop((target_height,target_width)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # using RGB operation
    ])

    valid_transform  = T.Compose([
        T.Resize((target_height,target_width)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Create train/valid datasets
    train_set = customDataset(root=opt.data,
                        split='train', transform=train_transform)
    valid_set = customDataset(root=opt.data,
                        split='test', transform=valid_transform)

    # Create train/valid loaders
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    test_loader = DataLoader(dataset=valid_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

    # Get images and labels in a mini-batch of train_loader
    # for image, label in train_loader:
    #     print('Size of image:', image.size())  # torch.Size([16, 3, 128, 256])
    #     print('Type of image:', image.dtype)   # float32
    #     print('Size of label:', label.size())  # batch_size   torch.Size([16])
    #     print('Type of label:', label.dtype)   # int64(long)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
