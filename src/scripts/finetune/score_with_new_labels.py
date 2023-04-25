import argparse
from collections import OrderedDict
import json
from pathlib import Path

import pytorch_lightning as pl
from pycrumbs import tracked
import torch
from torchvision import models

from model_drift.data import padchest, chexpert, mgb_data
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
    orig_state_dict = torch.load(model_path, map_location='cuda')
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


@tracked(directory_parameter="output_dir", require_empty_directory=True)
def score_with_new_labels(
        output_dir: Path,
        args: argparse.Namespace,
):
    train_labels = TRAINED_LABELS[args.trained_label_set]

    if train_labels == 'padchest':
        # HACKS
        train_labels[train_labels.index("Lesion")] = "Lung Lesion"
        # Should definitely fix this at some point...
        train_labels[train_labels.index("Pleural Abnormalities")] = "Pneumothorax"

    print("Train labels")
    print(train_labels)
    print("New Labels")
    print(list(mgb_data.LABEL_GROUPINGS.keys()))

    # Figure out the label mapping
    label_mapping = []
    for lab in mgb_data.LABEL_GROUPINGS.keys():
        try:
            orig_index = train_labels.index(lab)
        except ValueError as e:
            raise RuntimeError(
                f"Label {lab} not found in training labels"
            ) from e
        label_mapping.append(orig_index)

    model = load_model(args.model_path, num_classes=len(train_labels))
    model = model.cuda()
    model.eval()

    transformer = VisionTransformer.from_argparse_args(args)
    dm = MGBCXRDataModule.from_argparse_args(
        args,
        transforms=transformer.train_transform
    )
    print('Loading dataset')
    dm.load_datasets()
    print('Done Loading dataset')

    with torch.no_grad():
        for b, batch in enumerate(dm.test_dataloader()):
            print("Batch", b)

            im = batch['image'].cuda()
            prediction = model(im)
            activation = torch.sigmoid(prediction).cpu().numpy()
            prediction = prediction.cpu().numpy()

            for pred, act, lab, ind in zip(
                    prediction,
                    activation,
                    batch['label'].numpy(),
                    batch['index'],
            ):
                # Map the activations and predictions from the original set
                # of training labels to the new set
                mapped_predictions = pred[label_mapping]
                mapped_activations = act[label_mapping]

                image_results = {
                    "index": ind,
                    "score": mapped_predictions.tolist(),
                    "activation": mapped_activations.tolist(),
                    "label": lab.tolist(),
                }
                results_line = json.dumps(image_results)

                with output_dir.joinpath("preds.jsonl").open("a") as of:
                    print(results_line, file=of)

    print("Done")


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
