import os
import pathlib
import sys

import click
import numpy as np
import tifffile as tiff
import torch
import yaml
from rich import traceback, print

from models.unet import U2NET

WD = os.path.dirname(__file__)


@click.command()
@click.option('-i', '--input', required=True, type=str, help='Path to data file to predict.')
@click.option('-m', '--model', type=str,
              help='Path to an already trained XGBoost model. If not passed a default model will be loaded.')
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-o', '--output', type=str, help='Path to write the output to')
def main(input: str, model: str, cuda: bool, output: str):
    """Command-line interface for rts_package"""

    print(r"""[bold blue]
        rts_package
        """)

    print('[bold blue]Run [green]rts_package --help [blue]for an overview of all commands\n')
    if not model:
        model = get_pytorch_model(f'{WD}/models/model.ckpt')
    else:
        model = get_pytorch_model(model)
    if cuda:
        model.cuda()
    print('[bold blue] Parsing data')
    data_to_predict = read_data_to_predict(input)
    print('[bold blue] Performing predictions')
    predictions = predict(data_to_predict, model)
    print(predictions)
    if output:
        print(f'[bold blue]Writing predictions to {output}')
        write_results(predictions, output)


def read_data_to_predict(path_to_data_to_predict: str):
    """
    Parses the data to predict and returns a full Dataset include the DMatrix
    :param path_to_data_to_predict: Path to the data on which predictions should be performed on
    """
    return tiff.imread(path_to_data_to_predict)


def write_results(predictions: np.ndarray, path_to_write_to) -> None:
    """
    Writes the predictions into a human readable file.
    :param predictions: Predictions as a numpy array
    :param path_to_write_to: Path to write the predictions to
    """
    os.makedirs(pathlib.Path(path_to_write_to).parent.absolute(), exist_ok=True)
    np.save(path_to_write_to,predictions)
    pass


def get_pytorch_model(path_to_pytorch_model: str):
    """
    Fetches the model of choice and creates a booster from it.
    :param path_to_pytorch_model: Path to the xgboost model1
    """
    with open("hparams.yaml", 'r') as f:
        model_yaml = yaml.load(f, Loader=yaml.FullLoader)
    model = U2NET.load_from_checkpoint(path_to_pytorch_model, num_classes=5,
                                       len_test_set=120, **model_yaml).to('cpu')
    model.eval()
    return model


def predict(data_to_predict, model):
    img = data_to_predict[0, :, :]
    img = torch.from_numpy(np.expand_dims(np.expand_dims(img, 0), 0)).float()
    logits = model(img)[0]
    prediction = torch.argmax(logits.squeeze(), dim=0).cpu().detach().numpy().squeeze()
    return prediction


if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover
