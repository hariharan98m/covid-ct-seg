import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
from pathlib import Path
import pdb
import pandas as pd

def load_nii_file(scan_path, mask_path):
    nifi = nib.load(scan_path)
    scan = np.array(nifi.get_fdata())
    mask = np.array(nib.load(mask_path).get_fdata())
    num_slices = scan.shape[-1]
    # custom function to unstack tensor to image slices.
    unstack = lambda x: np.split(x, indices_or_sections=num_slices, axis=-1)
    # correct the view for human eye.
    scan, mask = np.transpose(scan, [1, 0, 2]), np.transpose(mask, [1, 0, 2])
    # return scan and mask
    return list(zip(unstack(scan), unstack(mask))), str(nifi.get_data_dtype())

def filter_with_lesions(samples):
    uniq_vals = lambda x: len(np.unique(x[1])) == 2
    return list(filter(uniq_vals, samples))

def save_samples(samples, filename, scan_save_path, mask_save_path):
    for i, (scan_pil, mask_pil) in enumerate(samples):
        scan_pil.save(str(scan_save_path / (filename + str(i) + '.jpg')))
        mask_pil.save(str(mask_save_path / (filename + str(i) + '.jpg')))

makepath = lambda path: Path.mkdir(path, exist_ok=True, parents=True)

def alter_range(data, newmin, newmax):
    min = np.min(data)
    max = np.max(data)

    norm_data = (data - min)/ (max - min) * (newmax - newmin) + newmin

    return norm_data

class JunMa:
    def __init__(self, metadata_file, rootdir):
        self.metadata = pd.read_csv(metadata_file)
        self.rootdir = rootdir

    def process_data(self, output_dir= 'processed_data/jun_ma'):
        scan_save_path = Path(output_dir) / 'scans'
        mask_save_path = Path(output_dir) / 'masks'

        # ensure these paths exist
        makepath(scan_save_path)
        makepath(mask_save_path)

        # headers: ct_scan, lung_mask, infection_mask, lung_and_infection_mask
        ct_scans = self.rootdir + '/ct_scans/' + self.metadata['ct_scan'].apply(lambda x: x.split('/')[-1])
        masks = self.rootdir + '/infection_mask/' + self.metadata['infection_mask'].apply(lambda x: x.split('/')[-1])

        for scan_path, mask_path in zip(ct_scans, masks):
            # in case files arent there, exit.
            if not Path(scan_path).is_file():
                continue

            filename = Path(scan_path).stem
            try:
                nii_samples, dtype = load_nii_file(scan_path, mask_path)
            except OSError as err:
                print(filename, ': ', err)
                continue

            print('junma: ', len(nii_samples), 'Dims: ', nii_samples[0][0].shape)
            # apply filter if needed
            if filter:
                nii_samples = filter_with_lesions(nii_samples)

            nii_pil_samples = []
            for scan, mask in nii_samples:
                if dtype == 'int16':
                    scan_pil = Image.fromarray((alter_range(scan[:, :, 0], 0, 255)).astype(np.uint8))
                    # enhance contrast of scan pil
                    scan_pil = ImageEnhance.Contrast(scan_pil).enhance(3.0)
                else:
                    scan_pil = Image.fromarray(scan[:, :, 0].astype(np.uint8))
                # just take the mask as such.
                mask_pil = Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8))
                nii_pil_samples.append((scan_pil, mask_pil))

            # save PIL samples
            # save_samples(nii_pil_samples, filename, scan_save_path, mask_save_path)


class Mosmed:
    def __init__(self, metadata_file, rootdir):
        self.metadata = pd.read_excel(metadata_file)
        self.rootdir = rootdir

    def process_data(self, output_dir):
        scan_save_path = Path(output_dir) / 'scans'
        mask_save_path = Path(output_dir) / 'masks'

        # ensure these paths exist
        makepath(scan_save_path)
        makepath(mask_save_path)

        samples_with_mask = self.metadata[~self.metadata['mask_file'].isna()]

        samples_with_mask['study_file'] = self.rootdir + samples_with_mask['study_file']
        samples_with_mask['mask_file'] = self.rootdir + samples_with_mask['mask_file']

        # headers: study_file, mask_file
        for scan_path, mask_path in zip(samples_with_mask['study_file'], samples_with_mask['mask_file']):
            # check if file is there or not.
            if not Path(scan_path).is_file():
                continue

            filename = Path(scan_path).stem
            nii_samples = load_nii_file(scan_path, mask_path)

            print('mosmed: ', len(nii_samples[0]), 'Dims: ', nii_samples[0][0][0].shape)
            # apply filter if needed
            if filter:
                nii_samples = filter_with_lesions(nii_samples)

            nii_pil_samples = []
            for scan, mask in nii_samples:
                scan_pil = Image.fromarray((alter_range(scan[:, :, 0], 0, 255)).astype(np.uint8))
                # enhance contrast of scan pil
                scan_pil = ImageEnhance.Contrast(scan_pil).enhance(2.5)
                # just take the mask as such.
                mask_pil = Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8))
                nii_pil_samples.append((scan_pil, mask_pil))

            # save samples
            # save_samples(nii_pil_samples, filename, scan_save_path, mask_save_path)

'''
vshshv3@vm-2:~/Downloads/covid19_ct/jun_ma$ ls
ct_scans  infection_mask  lung_and_infection_mask  lung_mask  metadata.csv
vshshv3@vm-2:~/Downloads/covid19_ct/jun_ma$ cd


vshshv3@vm-2:~/Downloads/covid19_ct/mosmed/masks$ cd ..
vshshv3@vm-2:~/Downloads/covid19_ct/mosmed$ ls
LICENSE  README_EN.md  README_EN.pdf  README_RU.md  README_RU.pdf  dataset_registry.xlsx  masks  studies
vshshv3@vm-2:~/Downloads/covid19_ct/mosmed$

'''

if __name__ == '__main__':
    jun_ma = JunMa('datasets/jun_ma/metadata.csv', 'datasets/jun_ma')
    # jun_ma.process_data(output_dir='processed_data_proper')

    mosmed = Mosmed('datasets/mosmed/dataset_registry.xlsx', 'datasets/mosmed')
    mosmed.process_data(output_dir='processed_data_proper')
