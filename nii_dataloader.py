import nibabel as nib
import matplotlib.pyplot as plt
import random
plt.style.use('dark_background')
import pdb
import numpy as np

ct_scan = 'datasets/jun_ma/ct_scans/radiopaedia_org_covid-19-pneumonia-14_85914_0-dcm.nii'
lung_mask = 'datasets/jun_ma/lung_mask/radiopaedia_14_85914_0.nii'
infection_mask = 'datasets/jun_ma/infection_mask/radiopaedia_14_85914_0.nii'

def load_nii_image():
    scan = nib.load(ct_scan).get_fdata()
    lung = nib.load(lung_mask).get_fdata()
    inf = nib.load(infection_mask).get_fdata()

    fig, ax = plt.subplots(3, 10, figsize = (30, 7))
    random.seed(0); random_indices = random.sample(range(110), 10)
    for i in range(10):
        # CT Scan
        ax[0, i].imshow(np.transpose(scan[:, :, random_indices[i]], [1, 0]) , cmap='gray')
        ax[0, i].set_title('SCAN %d' % (i+1))
        ax[0, i].axis('off')

        # Lung Mask
        ax[1, i].imshow(np.transpose(lung[:, :, random_indices[i]], [1, 0]))
        ax[1, i].set_title('Lung Mask %d' % (i+1))
        ax[1, i].axis('off')

        # Infection Mask
        ax[2, i].imshow(np.transpose(inf[:, :, random_indices[i]], [1, 0]))
        ax[2, i].set_title('Inf Mask %d' % (i+1))
        ax[2, i].axis('off')

        print('unique mask:', np.unique(lung))
        print('unique infection:', np.unique(inf))

    plt.savefig('file.png')

def load_mosmed():
    mask = 'datasets/mosmed/study_0303_mask.nii.gz'
    mask = nib.load(mask).get_fdata()

    fig, ax = plt.subplots(2, 10, figsize=(30, 7))
    random.seed(0); random_indices = random.sample(range(43), 10)

    for i in range(43):
        print(np.unique(mask[:,:,i]))

    for i in range(10):
        ax[i].imshow(np.transpose(mask[:, :, random_indices[i]], [1, 0]), cmap='gray')
        ax[i].set_title('SCAN %d' % (i + 1))
        ax[i].axis('off')

        print('unique mask:', np.unique(mask[:,:,i]))

    plt.savefig('masks.png')

import nibabel as nib
from pathlib import Path
def get_total_sample_count(directory = Path('Downloads/mosmed/COVID19_1110/studies/CT-0')):
    sample_cnt = 0
    for file in ['Downloads/mosmed/COVID19_1110/studies/CT-1/study_0255.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0256.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0257.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0258.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0259.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0260.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0261.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0262.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0263.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0264.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0265.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0266.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0267.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0268.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0269.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0270.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0271.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0272.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0273.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0274.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0275.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0276.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0277.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0278.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0279.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0280.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0281.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0282.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0283.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0284.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0285.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0286.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0287.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0288.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0289.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0290.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0291.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0292.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0293.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0294.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0295.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0296.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0297.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0298.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0299.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0300.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0301.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0302.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0303.nii.gz', 'Downloads/mosmed/COVID19_1110/studies/CT-1/study_0304.nii.gz']: # directory.rglob('*.nii'):
        try:
            scan = nib.load(file).get_fdata()
            sample_cnt += scan.shape[-1]
        except:
            print (file.name , ' is damaged')
    return sample_cnt

# print(get_total_sample_count())

# load_nii_image()

if __name__ == '__main__':
    load_mosmed()