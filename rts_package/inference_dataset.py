import argparse
import glob
import os.path
import pathlib
from argparse import ArgumentParser
from typing import List

import cv2
import numpy as np
import pytorch_lightning as pl
import tifffile as tiff
import torch
import yaml
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr._utils import visualization as viz
from matplotlib import pyplot, pyplot as plt
from torch import nn
from tqdm import tqdm
import seaborn as sns
from models.unet import UNET, U2NET
from utils import label2rgb


def main():
    parser = ArgumentParser(description='PyTorch Autolog Mnist Example')
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path to Images, if path is a folder all images from the folder are evaluated.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to Image, if path is a folder all images from the folder are evaluated.'
    )
    # TODO: Check if necessary or not.
    parser.add_argument(
        '--hparams',
        type=str,
        required=True,
        help='Path to hparams'
    )
    parser.add_argument('--interpretability', default=True, action=argparse.BooleanOptionalAction)
    args = vars(parser.parse_args())

    with open("hparams.yaml", 'r') as f:
        model_yaml = yaml.load(f, Loader=yaml.FullLoader)
    model = U2NET.load_from_checkpoint('/vol/data/u2net_0408.ckpt', num_classes=5,
                                      len_test_set=120, **model_yaml, strict=False).to('cpu')
    model.eval()
    classes = ['background', 'root', 'early elongation zone', 'late elongation zone', 'meristematic zone']
    imgs = glob.glob(
        os.path.join("/home/ubuntu/root-tissue-segmentation/root_tissue_segmentation/dataset/PHDFM/processed", "*.pt"))
    data_mask = torch.load(imgs[1])
    path = "/vol/data/labeled_img/"
    data = data_mask[:, :, :, 0:1].cpu().detach().numpy().squeeze()
    targets = data_mask[:, :, :, 1:2].cpu().detach().numpy().squeeze()
    os.makedirs(path, exist_ok=True)
    for i, (img, target) in tqdm(enumerate(zip(data, targets))):
        img_path = os.path.join(path, str(i))
        os.makedirs(img_path, exist_ok=True)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(int)

        labeled_img = label2rgb(0.7, rgb_img, target).cpu().detach().numpy().squeeze()
        cv2.imwrite(os.path.join(img_path, "img.png"), img)
        cv2.imwrite(os.path.join(img_path, "gt.png"), labeled_img.transpose(1, 2, 0))
        normalized_img = img.squeeze()

        normalized_img = torch.from_numpy(np.expand_dims(np.expand_dims(normalized_img, 0), 0)).float()
        pred = predict(img, model)
        labeled_img = label2rgb(0.7, rgb_img, pred).cpu().detach().numpy().squeeze()
        cv2.imwrite(os.path.join(img_path, "pred.png"), labeled_img.transpose(1, 2, 0))
        conv_layer = model.stage6
        target_classes = np.unique(pred).tolist()
        for target_class in target_classes:
            plot = grad_cam(model, normalized_img, conv_layer, target_class,
                            torch.from_numpy(pred[None, :][None, :].astype("int64")),
                            np.stack((img.squeeze(),) * 3, axis=-1))
            plt.tight_layout()
            sns.heatmap(plot, cmap="rainbow", xticklabels=False, yticklabels=False)
            plt.plot()
            plt.tight_layout()
            plt.savefig(os.path.join(img_path, f'{classes[target_class]}.png').replace(" ","_"),dpi=600,
                            pad_inches=0, bbox_inches='tight')
            plt.close()


def add_prediction_to_file(model, img_path, output_path):
    img = tiff.imread(img_path)
    prediction = predict(img[0, :, :], model)
    tif_img = np.concatenate((img, np.expand_dims(prediction, axis=0).astype('float32')), axis=0)
    os.makedirs(pathlib.Path(output_path).parent.absolute(), exist_ok=True)
    tiff.imwrite(output_path, tif_img, imagej=True)
    return prediction


def normalize(img, mean=0.6993, std=0.4158):
    img = img / 255.0
    img = img - mean
    return img / std


def add_ratio(img_path, ratio_folder="ratio_combined"):
    """
    Combines image and given ratiomeric image into one single image.
    :param img_path: Path to img/directory.
    :return: None
    """
    if isinstance(img_path, str) and os.path.isdir(img_path):
        return add_ratio(list(pathlib.Path(img_path).glob('*.tif')))
    elif isinstance(img_path, list):
        add_ratio(str(img_path.pop()))
        if img_path:
            return add_ratio(img_path)
        else:
            return
    elif os.path.isfile(img_path):
        tif_img = tiff.imread(img_path)
        os.makedirs(img_path.replace(f"input/{os.path.basename(img_path)}", ratio_folder), exist_ok=True)
        ratio_img = tiff.imread(img_path.replace("input", "output").replace(".tif", "_ratio.tif")).squeeze()
        ratio_img = np.nan_to_num(ratio_img, 0)
        if tif_img.shape[0] == 4:
            tif_img = np.concatenate((tif_img.astype('float32'), np.expand_dims(ratio_img, axis=0).astype('float32')),
                                     axis=0).astype('float32')
            tiff.imwrite(img_path.replace("input", ratio_folder), tif_img, imagej=True)
        return tif_img


def predict(img, model):
    #img = normalize(img)
    img = torch.from_numpy(np.expand_dims(np.expand_dims(img, 0), 0)).float()
    logits = model(img)[0]
    prediction = torch.argmax(logits.squeeze(), dim=0).cpu().detach().numpy().squeeze()
    return prediction


def grad_cam(model: pl.LightningModule, normalized_img: np.array, conv_layer: nn.Conv2d, target_class: int,
             prediction: torch.Tensor, image) -> List[pyplot.plot]:
    """
    Creates model interpretability.
    :param model: Trained PyTorch for multi-class semantic segmentation
    :param normalized_img: Normalized input for the model.
    :param prediction: argmax of the output.
    :param conv_layer: Convolutional Layer used for the evaluation.
    :param target_class: Class for gradCam.
    :return: Python plot containing the interpreted images.
    """

    def agg_segmentation_wrapper(inp):
        model_out = model(inp)[0]
        selected_indices = torch.zeros_like(model_out[0:1]).scatter_(1, prediction, 1)
        return (model_out * selected_indices).sum(dim=(2, 3))

    normalized_img.requires_grad = True
    lgc = LayerGradCam(agg_segmentation_wrapper, conv_layer)
    gc_attr = lgc.attribute(normalized_img, target=target_class)
    gc_np = gc_attr.cpu().detach().numpy().squeeze()
    max = np.max(gc_np)
    for i in range(gc_np.shape[0]):
        for j in range(gc_np.shape[0]):
            if gc_np[i,j]>=0:
                gc_np[i,j] = gc_np[i,j]/max
            else:
                gc_np[i, j] = 0
    upsample  = cv2.resize(gc_np, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    return upsample


def interpretability(img, model, prediction, path, classes):
    tif_img = tiff.imread(img)
    normalized_img = tif_img[0].squeeze()
    normalized_img = torch.from_numpy(np.expand_dims(np.expand_dims(normalized_img, 0), 0)).float()
    conv_layer = model.outconv
    target_classes = np.unique(prediction).tolist()
    os.makedirs(path, exist_ok=True)
    for target_class in target_classes:
        plot = grad_cam(model, normalized_img, conv_layer, target_class, prediction[None, :][None, :],
                        np.stack((tif_img[0].squeeze(),) * 3, axis=-1))
        plot[0].tight_layout()
        plot[0].savefig(os.path.join(path, f'{classes[target_class]}.png'),dpi=600,
                        pad_inches=0)
        plt.close()


if __name__ == "__main__":
    main()
