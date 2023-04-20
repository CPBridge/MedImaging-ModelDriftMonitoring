import argparse
from collections import OrderedDict
from pathlib import Path

import pytorch_lightning as pl
from pycrumbs import tracked
import torch
from torchvision import models

from model_drift.data import padchest, chexpert
from model_drift.data.datamodules import MGBCXRDataModule
from model_drift.data.transform import VisionTransformer


TRAINED_LABELS = {
    "padchest": list(padchest.LABEL_MAP.keys()),
    "chexpert": chexpert.LABELS,
}


def load_model(
        model_path: Path,
        num_classes: int,
):
    orig_state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = OrderedDict()

    # Somehow the state_dicts got pretty messed up... need to do some surgery
    if "model_state" in orig_state_dict:
        state_key = "model_state"
        prefixes = ["module.model."]
    elif "state_dict" in orig_state_dict:
        state_key = "state_dict"
        prefixes = ['model.model.', "model."]

    for k, v in orig_state_dict[state_key].items():
        for prefix in prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]  # remove prefix
                break
        new_state_dict[k] = v

    model = models.densenet121(pretrained=False, num_classes=14)
    model.load_state_dict(new_state_dict)

    return model


@tracked(directory_parameter="output_dir")
def score_with_new_labels(
        output_dir: Path,
        args: argparse.Namespace,
):
    train_labels = TRAINED_LABELS[args.trained_label_set]

    model = load_model(args.model_path, num_classes=len(train_labels))
    model.eval()

    transformer = VisionTransformer.from_argparse_args(args)
    dm = MGBCXRDataModule.from_argparse_args(args, transforms=transformer.train_transform)
    print('Loading dataset')
    dm.load_datasets()
    print('Done Loading dataset')

    for batch in dm.test_dataloader():
        print(batch[0])





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run a previously trained model on the MGB dataset and store the "
            "outputs for the classes that are used."
        )
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="path to model or registered model name",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="output directory",
        required=True,
    )
    parser.add_argument(
        "--trained_label_set",
        type=str,
        choices=TRAINED_LABELS.keys(),
        help="Labels that were used to train the model",
        required=True,
    )

    parser = MGBCXRDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VisionTransformer.add_argparse_args(parser)
    args = parser.parse_args()
    score_with_new_labels(args.output_dir, args)
