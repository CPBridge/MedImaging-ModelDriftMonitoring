#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import os
import pytorch_lightning as pl
import torch
# import yaml
from pathlib import Path
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from model_drift.callbacks import IOMonitor
from model_drift.models.finetune import CheXFinetune
from model_drift.azure_utils import download_model_azure, get_azure_logger
# from model_drift.data.datamodules import PadChestDataModule
from model_drift.data.datamodules import MGBCXRDataModule
from model_drift.data.transform import VisionTransformer
from model_drift.data.mgb_data import LABEL_GROUPINGS

from pycrumbs import tracked

num_gpus = torch.cuda.device_count()
num_cpus = os.cpu_count()
node_rank = os.environ.get("NODE_RANK", 0)
local_rank = os.environ.get("LOCAL_RANK", 0)
world_size = os.environ.get("WORLD_SIZE", 1)
global_rank = os.environ.get("RANK", 0)
rank_id = f"{node_rank}-{local_rank}"

print()
print("=" * 5)
print(" Pytorch Lightning Version:", pl.__version__)
print(" Pytorch Version:", torch.__version__)
print(" Num GPUs:", num_gpus)
print(" Num CPUs:", num_cpus)
print(" Node Rank:", node_rank)
print(" Local Rank:", local_rank)
print(" World Size:", world_size)
print(" Global Rank:", global_rank)
print(" Rank ID:", rank_id)
print("=" * 5)
print()

@tracked(directory_parameter="output_dir")
def main(output_dir: Path, args: argparse.Namespace) -> None:
    if args.num_workers < 0:
        args.num_workers = num_cpus

    args.weights_summary = "top" if rank_id == "0-0" else None
    output_dir = args.output_dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    args.gpus = num_gpus
    new_batch_size = max(args.batch_size, num_gpus * args.batch_size)
    if new_batch_size != args.batch_size:
        print(f"scaling batch size by device count {args.gpus} " f"from {args.batch_size} to {new_batch_size}")
        args.batch_size = new_batch_size

    model_dirpath = os.path.join(output_dir, "checkpoints")
    ckpt_path = os.path.join(model_dirpath, "last.ckpt")

    os.makedirs(model_dirpath, exist_ok=True)

    if os.path.exists(ckpt_path):
        args.resume_from_checkpoint = ckpt_path
        args.num_sanity_val_steps = 0

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dirpath,
        filename="{epoch:0>3}",
        save_last=True,
        save_top_k=-1,
        monitor="val/AUROC.mean",
        every_n_epochs=1,
    )

    #gpus=1 was hardcoded because argument didnt work and it speeds up debugging
    trainer = pl.Trainer.from_argparse_args(args, gpus=1)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(lr_monitor)
    trainer.callbacks.append(IOMonitor())

    # trainer.callbacks.append(GPUStatsMonitor())

    if args.run_azure:
        if args.pretrained:
            args.pretrained = download_model_azure(args.pretrained, args.output_dir)
        trainer.logger = get_azure_logger()

    transformer = VisionTransformer.from_argparse_args(args)
    # dm = PadChestDataModule.from_argparse_args(args, transforms=transformer.train_transform)
    dm = MGBCXRDataModule.from_argparse_args(args, train_transforms=transformer.train_transform, 
                                             val_transforms=transformer.val_transform,
                                             test_transforms=transformer.val_transform)
    args.num_classes = len(LABEL_GROUPINGS)
    params = vars(args)
    model = CheXFinetune.from_argparse_args(args, labels=dm.labels, params=params)

    # if rank_id == "0-0":
    #     with open(os.path.join(args.output_dir, "input.yaml"), 'w') as f:
    #         yaml.safe_dump(params, f)
    #     with open(os.path.join(args.output_dir, "model.txt"), 'w') as f:
    #         print(model, file=f)



    trainer.fit(model, dm)

    trainer.training_type_plugin.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_azure", type=int, dest="run_azure", help="run in AzureML", default="0")
    parser.add_argument("--output_dir", type=str, dest="output_dir", help="output directory", default="./outputs", )

    parser = VisionTransformer.add_argparse_args(parser)
    parser = CheXFinetune.add_model_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    # parser = PadChestDataModule.add_argparse_args(parser)
    parser = MGBCXRDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    main(args.output_dir, args)
