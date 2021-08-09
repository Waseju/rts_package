import argparse
import glob
import os.path
import pathlib
from argparse import ArgumentParser
from typing import List

import numpy as np
import pytorch_lightning as pl
import tifffile as tiff
import torch
import yaml
from captum.attr import LayerGradCam, LayerAttribution, GuidedBackprop, LRP
from captum.attr._utils import visualization as viz
from matplotlib import pyplot, pyplot as plt
from torch import nn
from tqdm import tqdm
from models.unet import U2NET


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
        required=False,
        help='Path to Image, if path is a folder all images from the folder are evaluated.'
    )
    # TODO: Check if necessary or not.
    parser.add_argument(
        '--hparams',
        type=str,
        required=False,
        help='Path to hparams'
    )
    parser.add_argument('--interpretability', default=True, action=argparse.BooleanOptionalAction)
    args = vars(parser.parse_args())

    with open("hparams.yaml", 'r') as f:
        model_yaml = yaml.load(f, Loader=yaml.FullLoader)
    model = U2NET.load_from_checkpoint('/vol/data/u2net_0408.ckpt', num_classes=5,
                                       len_test_set=120, **model_yaml).to('cpu')
    model.eval()
    imgs = glob.glob(f'{args["images"]}*/output/*.tif')
    classes = ['background', 'root', 'early elongation zone', 'late elongation zone', 'meristematic zone']

    for img in tqdm(imgs):
        img = img.replace("output", "input").replace("_ratio", "")
        add_ratio(img)

        prediction = add_prediction_to_file(model, img.replace("input", "ratio_combined"),
                                            img.replace("input", "final"))
        if False:
            interpret_path = img.replace("input", "interpret").replace(".tif", "")
            interpretability(img, model=model, prediction=torch.from_numpy(prediction),
                             path=interpret_path, classes=classes)


def add_prediction_to_file(model, img_path, output_path):
    img = tiff.imread(img_path)
    prediction = predict(img[0, :, :], model)
    tif_img = np.concatenate((img, np.expand_dims(prediction, axis=0).astype('float32')), axis=0)
    os.makedirs(pathlib.Path(output_path).parent.absolute(), exist_ok=True)
    tiff.imwrite(output_path, tif_img, imagej=True)
    return prediction


def normalize(img, mean=0.6993, std=0.4158):
    return img
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
        model_out = torch.unsqueeze(model(inp),0)
        print(model_out.shape)
        selected_indices = torch.zeros_like(model_out[0:1]).scatter_(1, prediction, 1)
        return (model_out * selected_indices).sum(dim=(2, 3))

    normalized_img.requires_grad = True
    lgc = LayerGradCam(agg_segmentation_wrapper, conv_layer)
    lrp = LRP(model)
    gc_attr = lrp.attribute(normalized_img)
    #gc_attr = lgc.attribute(normalized_img, target=target_class)
    upsampled_gc_attr = LayerAttribution.interpolate(gc_attr, normalized_img.shape[2:])
    return viz.visualize_image_attr(upsampled_gc_attr[0].cpu().permute(1, 2, 0).detach().numpy(),
                                    original_image=image,
                                    alpha_overlay=0.7,
                                    show_colorbar=True,
                                    sign="all",
                                    method="blended_heat_map")


def interpretability(img, model, prediction, path, classes):
    tif_img = tiff.imread(img)
    normalized_img = normalize(tif_img[0].squeeze())
    normalized_img = torch.from_numpy(np.expand_dims(np.expand_dims(normalized_img, 0), 0)).float()
    conv_layer = model.outconv
    target_classes = np.unique(prediction).tolist()
    os.makedirs(path, exist_ok=True)
    for target_class in target_classes:
        plot = grad_cam(model, normalized_img, conv_layer, target_class, prediction[None, :][None, :],
                        np.stack((tif_img[0].squeeze(),) * 3, axis=-1))
        plot[0].tight_layout()
        plot[0].savefig(os.path.join(path, f'{classes[target_class]}.pdf'),
                        pad_inches=0)
        plt.close()


if __name__ == "__main__":
    main()
