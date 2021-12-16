from pathlib import Path
import torchvision
import torch
import time
import numpy as np
import pdb
from dataset_creation import get_train_val_test_splits, COVID19_CT_dataset
from model import Model, model_train

train_transform = torchvision.transforms.Compose([
        # torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(p=1.0),
        torchvision.transforms.RandomAffine(10, translate=(0.1, 0.1), scale=None, shear=5, resample=False, fillcolor=0),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.543463], [0.36]),
    ])

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.543463], [0.36]),
])


if __name__== '__main__':
    classes = ['covid', 'non-covid']
    class_paths = {
        'covid': Path('datasets/san_diego_dataset/CT_COVID'),
        'non-covid': Path('datasets/san_diego_dataset/CT_NonCOVID')
    }

    # train-validation and test samples
    train_val_test_samples = get_train_val_test_splits(classes, paths = class_paths, ratios = [0.85, 0.05, 0.10])
    all_samples = get_train_val_test_splits(classes, class_paths, None, True)

    covid19_dataset = COVID19_CT_dataset(all_samples['all_samples'], transforms=val_transform)
    covid19_dataset_train = COVID19_CT_dataset(train_val_test_samples['train'], transforms= train_transform)
    covid19_dataset_val = COVID19_CT_dataset(train_val_test_samples['val'], transforms=val_transform)
    covid19_dataset_test = COVID19_CT_dataset(train_val_test_samples['test'], transforms=val_transform)

    covid19_sampler = covid19_dataset_train.sampler
    balanced_dataloader_train = torch.utils.data.DataLoader(covid19_dataset_train, sampler=covid19_sampler, batch_size=128,
                                                      num_workers=50)
    dataloader_val = torch.utils.data.DataLoader(covid19_dataset_val, batch_size=128,
                                                            num_workers=50)
    dataloader_test = torch.utils.data.DataLoader(covid19_dataset_test, batch_size=128,
                                                          num_workers=50)

    start = time.time()
    means, stds = [], []
    for batch in balanced_dataloader_train:
        batch_data, labels = batch
        print('cls0: ', torch.sum(labels == 0).item(), 'cls1: ', torch.sum(labels == 1).item())
        mean, std = torch.mean(batch_data).item(), torch.std(batch_data).item()
        print('mean: ', mean, 'std: ', std)
        means.append(mean); stds.append(std);
    print('Average mean %f, average std: %.2f' % (np.mean(means), np.mean(stds)))
    end = time.time()
    # pdb.set_trace()

    model = Model(2)
    model_train(balanced_dataloader_train, dataloader_val, model, epochs = 30)