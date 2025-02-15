#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import os
from argparse import Namespace
from pathlib import Path
from pycrumbs import tracked

import pytorch_lightning as pl
import torch

library_path = str(Path(__file__).parent.parent.parent)
PYPATH = os.environ.get("PYTHONPATH", "").split(":")
if library_path not in PYPATH:
    PYPATH.append(library_path)
    os.environ["PYTHONPATH"] = ":".join(PYPATH)

import model_drift.azure_utils
from model_drift import helpers
from model_drift.models.resnet_features import Resnet_Features
from model_drift.data.datamodules import (
    PadChestDataModule,
    PediatricCheXpertDataModule,
    MIDRCDataModule,
    MGBCXRDataModule,
)
from model_drift.callbacks import ResNetFeaturePredictionWriter
from model_drift.data.transform import VisionTransformer


# Add your data module here. Two examples are:
data_modules = {
    "padchest": PadChestDataModule,
    "peds": PediatricCheXpertDataModule,
    "midrc": MIDRCDataModule,
    "mgb": MGBCXRDataModule,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, dest="output_dir", help="output_dir", default="outputs")

    parser.add_argument("--dataset", type=str, dest="dataset", help="dataset", choices=list(data_modules),
                        default='padchest')
    temp_args, _ = parser.parse_known_args()
    #dm_cls = data_modules[temp_args.dataset]
    #parser = dm_cls.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MGBCXRDataModule.add_argparse_args(parser)
    parser = VisionTransformer.add_argparse_args(parser)
    args = parser.parse_args()
    score(output_dir=args.output_dir, args=args)


@tracked(directory_parameter="output_dir")
def score(output_dir: str, args):

    helpers.basic_logging()

    ddp_model_check_key = "_LOCAL_MODEL_PATH_"

    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    node_rank = os.environ.get("NODE_RANK", 0)
    local_rank = os.environ.get("LOCAL_RANK", 0)

    print()
    print("=" * 5)
    print(" Pytorch Lightning Version:", pl.__version__)
    print(" Pytorch Version:", torch.__version__)
    print(" Num GPUs:", num_gpus)
    print(" Num CPUs:", num_cpus)
    print(" Node Rank:", node_rank)
    print(" Local Rank:", local_rank)
    print("=" * 5)
    print()

    args.gpus = num_gpus
    args.output_dir = args.output_dir.replace("//", "/")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    args.default_root_dir = args.output_dir

    model = Resnet_Features()
    transformer = VisionTransformer.from_argparse_args(args)

    #dm_cls = data_modules[args.dataset]
    #dm = dm_cls.from_argparse_args(args, transforms=transformer.val_transform)
    dm = MGBCXRDataModule.from_argparse_args(args, transforms=transformer.val_transform)
    writer = ResNetFeaturePredictionWriter(args.output_dir)

    if args.num_workers < 0:
        args.num_workers = num_cpus

    model.eval()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(writer)
    _ = trainer.predict(model, dm)

    trainer.training_type_plugin.barrier()

    writer.merge_prediction_files(trainer)


if __name__ == "__main__":
    main()
