import torch
import torch.nn as nn
from model import obtain_latest_checkpoint, AdditiveAttentionFCN, dice_dissim, ensure_exists, FCNEdgeDecoder, MultiResidualAttentionUpsamplerEdgeDecoder, LocalAttentionUpsamplingModelMultiRec, BaselineFCN, ensure_exists
from pathlib import Path
import torchvision
from ct_dataset import COVID19_CT_dataset, train_val_splits, HoriFlip, Affine
from std_models import Linknet, DeepLabV3, PSPNet, FCN_ResNet50
import random
import pdb
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def load_model(model, model_dir, file= None):
    latest_checkpoint_file = str(Path(model_dir) / file) if file is not None else obtain_latest_checkpoint(Path(model_dir))
    print('Loaded ', latest_checkpoint_file, '!')
    if latest_checkpoint_file is not None:
        checkpoint = torch.load(str(latest_checkpoint_file), map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

def dice_samplewise(inputs, targets):
    # b x 256 x 256
    smooth = 10e-6
    # flatten label and prediction tensors
    intersection = (inputs * targets).sum((1,2))   # b x 1
    sums = inputs.sum((1,2)) + targets.sum((1,2))   # b x 1

    dice = (2. * intersection + smooth) / (sums + smooth)

    return dice


SMOOTH = 1e-6
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch

def iou_samplewise(inputs, targets):
    # flatten label and prediction tensors
    smooth = .1
    # flatten label and prediction tensors
    intersection = (inputs * targets).sum((1, 2))  # b x 1
    union = inputs.sum((1,2)) + targets.sum((1,2)) - intersection  # b x 1

    iou = (intersection + smooth) / (union + smooth)

    return iou   #  b x 1

def iou_all(inputs, targets):
    # flatten label and prediction tensors
    smooth = .1
    # flatten label and prediction tensors
    intersection = (inputs * targets).sum()  # b x 1
    union = inputs.sum() + targets.sum() - intersection  # b x 1
    iou = (intersection + smooth) / ((union + smooth))
    return iou


mae_loss = torch.nn.L1Loss(reduction='none')

def get_metrics_samplewise(actual, predicted, probs):
    '''
        actual and predicted are tensors of 0 and 1.  #  b x 256 x 256
    '''
    dice = dice_samplewise(actual, predicted)

    # a, p = actual.numpy(), predicted.numpy()
    # intersection = np.logical_and(a[0], p[0])
    # union = np.logical_or(a[0], p[0])
    # iou_score = np.sum(intersection) / np.sum(union)
    # print('iou_score:', iou_score)
    smooth = 10e-9

    tp = torch.where((actual == 1.) * (predicted ==1.), torch.ones_like(actual), torch.zeros_like(actual)).sum((1,2))
    fp = torch.where((actual == 0.) * (predicted ==1.), torch.ones_like(actual), torch.zeros_like(actual)).sum((1,2))
    tn = torch.where((actual == 0.) * (predicted ==0.), torch.ones_like(actual), torch.zeros_like(actual)).sum((1,2))
    fn = torch.where((actual == 1.) * (predicted ==0.), torch.ones_like(actual), torch.zeros_like(actual)).sum((1,2))

    specificity = tn / (tn+fp + smooth)
    precision = tp / (tp+fp + smooth)
    sensitivity = tp / (tp + fn + smooth)

    iou = iou_samplewise(actual, predicted)
    mae = mae_loss(actual, predicted).sum((1,2)) / (256. * 256.)

    batch_size, spatial_h, spatial_w = actual.shape
    actual.unsqueeze_(dim=-1)
    act = torch.zeros(batch_size, spatial_h, spatial_w, 2)
    act.scatter_(-1, actual.long(), 1.0)

    return {
        'dice': dice.numpy().tolist(),
        'iou': iou.numpy().tolist(),
        'specificity': specificity.numpy().tolist(),
        'precision': precision.numpy().tolist(),
        'sensitivity': sensitivity.numpy().tolist(),
        'mae': mae.numpy().tolist(),
        'actual': act.view(-1, 2).numpy().tolist(),
        'probs': probs.permute(0, 2, 3, 1).reshape(-1, 2).numpy().tolist(),    # (10, 2, 16, 16) -> (10, 16, 16, 2)
    }

def auc_vals(actual, probs):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = dict(), dict(), dict()
    n_classes = 2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(actual[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc

def generate_predictions(model, dataloader_samples, edges = False, deeplab_special = False):
    data, masks = dataloader_samples[0].to(device), dataloader_samples[1].to(device)
    if edges:
        edge_map = dataloader_samples[2].to(device)
    # model.train()
    with torch.no_grad():
        model_input = (data, edge_map) if edges else data
        preds = model(model_input) if not deeplab_special else model(model_input)['out']
        preds = preds.detach().cpu()
        softmax_preds = nn.functional.softmax(preds, dim=1)
        _, output_indices = torch.max(softmax_preds, dim=1, keepdim=False)   # b x 256 x 256
        output = output_indices.float()
    return data.cpu(), masks.float().cpu(), output, softmax_preds


def get_metrics_batchwise(actual, preds, probs, train = False, deeplab_special = False):
    dice = 1- dice_dissim(actual, preds)
    iou = iou_all(actual, preds)
    if train:
        return {
            'dice': dice,
            'iou': iou
        }

    smooth = 10e-9

    tp = torch.where((actual == 1.) * (preds ==1.), torch.ones_like(actual), torch.zeros_like(actual)).sum()
    fp = torch.where((actual == 0.) * (preds ==1.), torch.ones_like(actual), torch.zeros_like(actual)).sum()
    tn = torch.where((actual == 0.) * (preds ==0.), torch.ones_like(actual), torch.zeros_like(actual)).sum()
    fn = torch.where((actual == 1.) * (preds ==0.), torch.ones_like(actual), torch.zeros_like(actual)).sum()

    specificity = tn / (tn+fp + smooth)
    precision = tp / (tp+fp + smooth)
    sensitivity = tp / (tp + fn + smooth)

    mae = mae_loss(actual, preds).mean()

    batch_size, spatial_h, spatial_w = actual.shape
    actual.unsqueeze_(dim=-1)
    act = torch.zeros(batch_size, spatial_h, spatial_w, 2)
    act.scatter_(-1, actual.long(), 1.0)

    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'specificity': specificity.item(),
        'precision': precision.item(),
        'sensitivity': sensitivity.item(),
        'mae': mae.item(),
        'actual': act.view(-1, 2).numpy().tolist(),
        'probs': probs.permute(0, 2, 3, 1).reshape(-1, 2).numpy().tolist(),    # (10, 2, 16, 16) -> (10, 16, 16, 2)
    }


def compute_dataloader_metrics(model, dataloader, edges = False, samplewise = False, train = False, deeplab_special=False):
    metrics_values = {
        'dice': [], 'iou': []
    }
    if not train:
        metrics_values.update({
            'specificity': [],
            'sensitivity': [],
            'mae': [],
            'precision': [],
            'actual': [],
            'probs': []
        })
    for data in dataloader:
        _, actual, preds, probs = generate_predictions(model, data, edges, deeplab_special)
        m = get_metrics_samplewise(actual, preds, probs) if samplewise else get_metrics_batchwise(actual, preds, probs)
        for key in metrics_values:
            computed_val = [m[key]] if isinstance(m[key], float) else m[key]
            if key == 'dice':
                print('dice:', computed_val)
            metrics_values[key].extend(computed_val)
    # pdb.set_trace()
    if not train:
        auc = auc_vals(np.array(metrics_values['actual']), np.array(metrics_values['probs']))
        del metrics_values['actual']; del metrics_values['probs']
        metrics_values['auc_class0'] = auc[0]; metrics_values['auc_class1'] = auc[1]
    return metrics_values

from data_preprocessor import alter_range

def do_plot(data, actual, preds, model_name= 'random', save = False, batch_index = 0, dice_vals = None):
    figures_save_path = Path('ct_outputs/pred_figs') / model_name
    ensure_exists([figures_save_path / 'actual', figures_save_path / 'pred', figures_save_path / 'ct'])

    if save:
        for i in range(len(data)):
            pil_ct = Image.fromarray(alter_range(data[i, 0].numpy(), 0, 255.).astype(np.uint8))
            actual_mask = Image.fromarray((actual[i, :, :, 0].numpy() * 255.).astype(np.uint8))
            pred_mask = Image.fromarray((preds[i].numpy() * 255.).astype(np.uint8))

            pil_ct.save(str(figures_save_path / 'ct' / ('ct_dice%.3f_batch%d_%d.jpg' % (dice_vals[i], batch_index, i))))
            actual_mask.save(str(figures_save_path / 'actual' / ('actual_dice%.3f_batch%d_%d.jpg' % (dice_vals[i], batch_index, i))))
            pred_mask.save(str(figures_save_path / 'pred' / ('pred_dice%.3f_batch%d_%d.jpg' % (dice_vals[i], batch_index, i))))
    else:
        fig, ax = plt.subplots(3, 10, figsize = (30, 7))

        for i in range(10):
            # CT Scan
            ax[0, i].imshow(data[i, 0, :, :] , cmap='gray')
            ax[0, i].set_title('SCAN %d' % (i+1))
            ax[0, i].axis('off')

            # Actual Mask
            ax[1, i].imshow(actual[i, :, :, 0])
            ax[1, i].set_title('Actual Mask %d' % (i+1))
            ax[1, i].axis('off')

            # Predicted Mask
            ax[2, i].imshow(preds[i, :, :])
            ax[2, i].set_title('Predicted Mask %d' % (i+1))
            ax[2, i].axis('off')

        plt.savefig('%d.png' % model_name)

def save_predictions(model_name, save_path = 'predicted_samples'):
    ensure_exists([Path(save_path) / model_name])

def filter_lesion_size(dataset, threshold, batch_size):
    def large_lesion(batch):
        batch = list(filter(lambda x: x[1].sum()/(256*256)*100.>threshold , batch))
        print('LARGE: ',get_lesion_area(batch.copy()), '\n\n')
        return torch.utils.data.dataloader.default_collate(list(batch)) # torch.utils.data._utils.collate.default_collate(batch)

    def small_lesion(batch):
        def filter_it(x):
            fraction = x[1].sum() / (256 * 256) * 100.
            return True if fraction>0.45 and fraction <= threshold else False
        batch = list(filter(filter_it, batch))
        print('SMALL: ',get_lesion_area(batch.copy()), '\n\n')
        return torch.utils.data.dataloader.default_collate(list(batch))

    small_lesion_dataloader = torch.utils.data.DataLoader(dataset, batch_size, True, num_workers=50, collate_fn = small_lesion)
    large_lesion_dataloader = torch.utils.data.DataLoader(dataset, batch_size, True, num_workers=50, collate_fn = large_lesion)

    return small_lesion_dataloader, large_lesion_dataloader


def get_lesion_area(dataloader):
    lesion_sizes = []
    for data in dataloader:
        ct, mask = data[0], data[1]
        ls = (mask.sum((-2,-1))/(256*256)*100).numpy().tolist()
        lesion_sizes.append(ls)
    return lesion_sizes


def get_sample_batch(dataset, size=10):
    random.seed(0); random_indices = random.sample(range(100), size)

    subset = torch.utils.data.Subset(dataset, random_indices)
    return next(iter(torch.utils.data.DataLoader(subset, batch_size=size)))


def choose_vis_samples(dices, ranges):
    r1, r2, r3 = ranges
    chosen = []
    for i in range(len(dices)):
        d1, d2, d3 = dices[i]
        if d1>=r1[0] and d1<=r1[1]:
            if d2>=r2[0] and d2<=r2[1]:
                if d3>=r3[0] and d3<=r3[1]:
                    chosen.append(i)
    return chosen


transforms = torchvision.transforms.Compose([
    HoriFlip(0.5),
    Affine(translate_xy=(0.1, 0.1), shear_angle_range=(-5, 5), rotate_angle_range=(-10, 10))
])

split_data = train_val_splits(dataset_path='processed_data_proper')

ct_seg_dataset_train = COVID19_CT_dataset(samples=split_data['train'], scan_norm=(0.5330, 0.3477),
                                          transforms=transforms, edges=True)
ct_seg_dataset_val = COVID19_CT_dataset(samples=split_data['val'], scan_norm=(0.5330, 0.3477),
                                        transforms=None, edges=True)
ct_seg_dataset_test = COVID19_CT_dataset(samples=split_data['test'], scan_norm=(0.5330, 0.3477),
                                         transforms=None, edges=True)


# all 40 for fcn-edge-decoder
# all 55 for upsampler. 60 train
# big guy - 35
ct_seg_dataloader_train = torch.utils.data.DataLoader(ct_seg_dataset_train, batch_size=50, num_workers=50)
ct_seg_dataloader_val = torch.utils.data.DataLoader(ct_seg_dataset_val, batch_size=50, num_workers=50)  # no disturb batch size = 10
ct_seg_dataloader_test = torch.utils.data.DataLoader(ct_seg_dataset_test, batch_size=50, num_workers=50)
ngpu = 2


if __name__ == '__main__2':

    multi_residual_attn_upsampler_edge_decoder = load_model(MultiResidualAttentionUpsamplerEdgeDecoder(),
                                                            'ct_outputs/ct_models/MultiResAttnUpsamplingEdgeDecoder')
    fcn_edge_decoder = load_model(FCNEdgeDecoder(), 'ct_outputs/ct_models/FCNEdgeDecoder')
    local_attn_upsampling_multi_rec = load_model(LocalAttentionUpsamplingModelMultiRec(),
                                                 'ct_outputs/ct_models/LocalAttnUpsamplingMultiRec')

    multi_residual_attn_upsampler_edge_decoder = multi_residual_attn_upsampler_edge_decoder.to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        multi_residual_attn_upsampler_edge_decoder = nn.DataParallel(multi_residual_attn_upsampler_edge_decoder,
                                                                     list(range(ngpu)))
    dice1 = compute_dataloader_metrics(multi_residual_attn_upsampler_edge_decoder, ct_seg_dataloader_val, edges=True,
                                       samplewise=True, train=True)
    del multi_residual_attn_upsampler_edge_decoder

    fcn_edge_decoder = fcn_edge_decoder.to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        fcn_edge_decoder = nn.DataParallel(fcn_edge_decoder, list(range(ngpu)))
    dice2 = compute_dataloader_metrics(fcn_edge_decoder, ct_seg_dataloader_val, edges=True, samplewise=True, train=True)
    del fcn_edge_decoder

    local_attn_upsampling_multi_rec = local_attn_upsampling_multi_rec.to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        local_attn_upsampling_multi_rec = nn.DataParallel(local_attn_upsampling_multi_rec, list(range(ngpu)))
    dice3 = compute_dataloader_metrics(local_attn_upsampling_multi_rec, ct_seg_dataloader_val, edges=True,
                                       samplewise=True, train=True)
    del local_attn_upsampling_multi_rec

    ranges = [(0.73, 0.765), (0.77, 0.81), (0.83, 0.87)]
    indices = choose_vis_samples([dice1, dice2, dice3], ranges)
    pdb.set_trace()

if __name__ == '__main__':

    # multi_residual_attn_upsampler_edge_decoder = load_model(MultiResidualAttentionUpsamplerEdgeDecoder(), 'ct_outputs/ct_models/MultiResAttnUpsamplingEdgeDecoder')   # batch 45
    # fcn_edge_decoder = load_model(FCNEdgeDecoder(), 'ct_outputs/ct_models/FCNEdgeDecoder', 'epoch97.pth')    # batch 40
    local_attn_upsampling_multi_rec = load_model(LocalAttentionUpsamplingModelMultiRec(), 'ct_outputs/ct_models/LocalAttnUpsamplingMultiRec', None) #'epoch67.pth')
    # pspnet = load_model(PSPNet(), 'ct_outputs/ct_models/PSPNet')
    # deeplabv3 = load_model(DeepLabV3(), 'ct_outputs/ct_models/DeepLabV3')
    # lnet = load_model(Linknet(), 'ct_outputs/ct_models/LinkNet')
    # baseline_fcn = load_model(BaselineFCN(), 'ct_outputs/ct_models/SimpleBaseline')
    # additive_attn_fcn = load_model(AdditiveAttentionFCN(), 'ct_outputs/ct_models/FCNwithAdditiveAttention')
    # fcn = load_model(FCN_ResNet50(), 'ct_outputs/ct_models/FCN8s', 'epoch15.pth')   # batch 55
    # r2u_net = load_model(FCN_ResNet50(), 'ct_outputs/ct_models/FCN8s', 'epoch13.pth')   # batch 64

    model = local_attn_upsampling_multi_rec #multi_residual_attn_upsampler_edge_decoder
    model = model.to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, list(range(ngpu)))

    # small - large: batches
    #

    small_dl, large_dl = filter_lesion_size(dataset=ct_seg_dataset_test, threshold=0.744, batch_size=64)
    small = compute_dataloader_metrics(model, small_dl, edges=False, samplewise=False, train=True, deeplab_special=False)
    large = compute_dataloader_metrics(model, large_dl, edges=False, samplewise=False, train=True, deeplab_special=False)

    print('small: ', np.mean(small['dice']))
    print('large: ', np.mean(large['dice']))
    pdb.set_trace()
    exit()

    # to generate plots.
    # pdb.set_trace()
    # batch = get_sample_batch(ct_seg_dataset_val, size=10)
    # for i, batch in enumerate(ct_seg_dataloader_val):
    #     data, actual, preds, probs = generate_predictions(model, batch, edges = False)
    #     sample_met = get_metrics_samplewise(actual, preds, probs)
    #     print('sample DICE:', sample_met['dice'])
    #     # pdb.set_trace()
    #     do_plot(data, actual, preds, model_name = 'decoder2', save = True, batch_index = i, dice_vals = sample_met['dice'])
    # exit(0)

    '''Code to compute test-val-train metrics'''

    # BATCHWISE : computing metrics part of it
    metrics_values = compute_dataloader_metrics(model, ct_seg_dataloader_test, edges=True, samplewise=False, train=False, deeplab_special=False)
    # metric vals
    print('TEST: Batch-wise:')
    for m, items in metrics_values.items():
        print(m, ': %.4f' % (np.mean(items)*100 if m is not 'mae' else np.mean(items)))
    print('\n')
    # exit(0)

    # BATCHWISE : computing metrics part of it
    # pdb.set_trace()
    metrics_values = compute_dataloader_metrics(model, ct_seg_dataloader_val, edges=True, samplewise=False, train=False)
    print('VAL: Batch-wise:')
    for m, items in metrics_values.items():
        print(m, ': %.4f' % (np.mean(items) * 100 if m is not 'mae' else np.mean(items)))

    print('\n')

    # metric vals
    metrics_values = compute_dataloader_metrics(model, ct_seg_dataloader_train, edges=True,
                                                samplewise=False, train=True)
    print('TRAIN: Batch-wise:')
    for m, items in metrics_values.items():
        print(m, ': %.4f' % (np.mean(items) * 100 if m is not 'mae' else np.mean(items)))

    # print('Overall DICE: ', aux(ct_seg_dataloader_val, additive_attn_fcn))
# Image.fromarray((inputs[0].numpy() * 255.).astype(np.uint8)).save('input.jpg')