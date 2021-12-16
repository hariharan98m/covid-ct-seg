import random
import torch
import numpy as np
from PIL import Image as pil_image
import torchvision
import pdb
from pathlib import Path
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def read_pil_image(path, mask = False):
    inp_shape = [256, 256]
    resized = pil_image.open(path).resize(inp_shape)
    if mask:
        return resized.convert('L')
    return resized.convert('L')


'''
torchvision.transforms.RandomHorizontalFlip(p=1.0),
torchvision.transforms.RandomAffine(10, translate=(0.1, 0.1), scale=None, shear=5, resample=False, fillcolor=0),
torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize([0.4560], [0.26442]),
'''
import torchvision.transforms.functional as TF

class HoriFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, mask = sample

        if random.random() > self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

class Affine(object):
    def __init__(self, translate_xy, shear_angle_range, rotate_angle_range):
        self.tx, self.ty = translate_xy
        self.shear_low, self.shear_high = shear_angle_range
        self.rotate_low, self.rotate_high = rotate_angle_range

    def __call__(self, sample):
        image, mask = sample
        h, w = image.size

        # get random parameters
        shear = random.uniform(self.shear_low, self.shear_high)
        translate_x = random.uniform(0, self.tx) * w
        translate_y = random.uniform(0, self.ty) * h
        rotate = random.uniform(self.rotate_low, self.rotate_high)

        # print('shear:', shear, ' tx:', translate_x, ' ty:', translate_y, ' rot:', rotate)

        # do the operations
        image = TF.affine(image, angle=rotate, translate=[translate_x, translate_y], shear = shear, scale = 1)
        mask = TF.affine(mask, angle=rotate, translate=[translate_x, translate_y], shear = shear, scale=1)

        return image, mask


class COVID19_CT_dataset():
    def __init__(self, samples, scan_norm, transforms = None, edges = False):
        self.samples = samples
        self.transforms = transforms
        self.toTensorTransform = torchvision.transforms.ToTensor()
        self.norm_mean, self.norm_std = scan_norm
        self.edges = edges

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        scan = read_pil_image(self.samples[item][0])
        mask = read_pil_image(self.samples[item][1], mask = True)

        # np.stack(scan, mask, axis=-1)

        transformed_scan, transformed_mask = scan, mask
        if self.transforms is not None:
            transformed_scan, transformed_mask = self.transforms((scan, mask))

        norm_transform = torchvision.transforms.Normalize([self.norm_mean], [self.norm_std])

        scan_mask_edges = [norm_transform(self.toTensorTransform(transformed_scan)),
                           torch.from_numpy(np.array(transformed_mask.convert('1')) * 1.0) ]

        if self.edges:
            edges = cv2.Canny(np.asarray(transformed_scan),200,220)
            scan_mask_edges.append(torch.from_numpy(edges.astype(np.float32)/255.).unsqueeze(dim=0))

        return scan_mask_edges #.unsqueeze(dim=0)

        # torch.from_numpy(np.array(scan)), torch.from_numpy(np.array(mask))

def train_val_splits(dataset_path, ratios= [0.85, 0.05, 0.10]):
    path = Path(dataset_path)
    img_dir = path / 'scans'
    mask_dir = path / 'masks'

    samples = sorted([(str(img_dir / p.name), str(mask_dir / p.name)) for p in img_dir.iterdir() if p.suffix == '.jpg'])    #[:1000]
    random.seed(0); random.shuffle(samples)

    train_marker = round(len(samples) * ratios[0])
    val_marker = train_marker + round(len(samples) * ratios[1])

    return {
        'train': samples[:train_marker],
        'val': samples[train_marker:val_marker],
        'test': samples[val_marker:]
    }

if __name__ == '__main__':
    transforms = torchvision.transforms.Compose([
        HoriFlip(0.5),
        Affine(translate_xy=(0.1, 0.1), shear_angle_range=(-5, 5), rotate_angle_range=(-10, 10))
    ])

    split_data = train_val_splits(dataset_path= 'processed_data_proper')

    ct_seg_dataset_train = COVID19_CT_dataset(samples=split_data['train'], scan_norm = (0.5330, 0.3477), transforms=transforms, edges=True)
    ct_seg_dataset_val = COVID19_CT_dataset(samples=split_data['val'], scan_norm = (0.5330, 0.3477), transforms=transforms, edges=True)

    ct_seg_dataloader_train = torch.utils.data.DataLoader(ct_seg_dataset_train, batch_size = 128, num_workers = 0)
    ct_seg_dataloader_val = torch.utils.data.DataLoader(ct_seg_dataset_train, batch_size=128, num_workers=0)

    for batch in ct_seg_dataloader_train:
        ct, mask, edges = batch
        print(torch.unique(edges))
        print('mean: ', torch.mean(ct), 'std: ', torch.std(ct))
        print(ct.shape, mask.shape)