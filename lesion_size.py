import torch
from ct_dataset import COVID19_CT_dataset, train_val_splits, HoriFlip, Affine
import torchvision
transforms = torchvision.transforms.Compose([
    HoriFlip(0.5),
    Affine(translate_xy=(0.1, 0.1), shear_angle_range=(-5, 5), rotate_angle_range=(-10, 10))
])
import numpy as np
import pdb
import matplotlib.pyplot as plt

split_data = train_val_splits(dataset_path='processed_data_proper')

ct_seg_dataset_train = COVID19_CT_dataset(samples=split_data['train'], scan_norm=(0.5330, 0.3477),
                                          transforms=transforms, edges=False)
ct_seg_dataset_val = COVID19_CT_dataset(samples=split_data['val'], scan_norm=(0.5330, 0.3477),
                                        transforms=None, edges=False)
ct_seg_dataset_test = COVID19_CT_dataset(samples=split_data['test'], scan_norm=(0.5330, 0.3477),
                                         transforms=None, edges=False)

# all 40 for fcn-edge-decoder
# all 55 for upsampler. 60 train
ct_seg_dataloader_train = torch.utils.data.DataLoader(ct_seg_dataset_train, batch_size=128, num_workers=50)
ct_seg_dataloader_val = torch.utils.data.DataLoader(ct_seg_dataset_val, batch_size=256, num_workers=50)
ct_seg_dataloader_test = torch.utils.data.DataLoader(ct_seg_dataset_test, batch_size=256, num_workers=50)
ngpu = 2


def get_lesion_area(dataloader):
    lesion_sizes = []
    for data in dataloader:
        ct, mask = data
        ls = (mask.sum((1,2))/(256*256)*100).numpy().tolist()
        lesion_sizes.extend(ls)
    return lesion_sizes

if __name__ == '__main__':
    lesion_areas = get_lesion_area(dataloader=ct_seg_dataloader_test)
    print(lesion_areas)
    print(np.median(lesion_areas))
    exit(0)

    plt.hist(lesion_areas)
    plt.savefig('lesions.png')
    plt.show()

    plt.boxplot(lesion_areas)
    plt.savefig('lesions_box.png')