#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import os
from pathlib import Path


import datetime
import pandas as pd
import pydicom
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

from model_drift import settings
from model_drift.data.dataset import (
    ChestXrayDataset,
    PediatricChestXrayDataset,
    MIDRCDataset,
    MGBCXRDataset,
)
from model_drift.data.padchest import PadChest, LABEL_MAP, BAD_FILES
from model_drift.data import mgb_data
from model_drift import mgb_locations


def _split_dates(s):
    if s is None:
        return tuple(settings.PADCHEST_SPLIT_DATES)
    try:
        return tuple([ss.strip() for ss in s.split(",")])
    except BaseException:
        raise argparse.ArgumentTypeError("Dates must be date1,date2")


class BaseDatamodule(pl.LightningDataModule):
    def __init__(self, data_folder,
                 transforms=None,

                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,

                 batch_size=32,
                 num_workers=-1,
                 train_kwargs=None,
                 val_kwargs=None,
                 test_kwargs=None,

                 output_dir='./',

                 frontal_only=False,
                 train_frontal_only=None,
                 val_frontal_only=None,
                 test_frontal_only=None,
                 ):
        super().__init__()
        if transforms is None and (train_transforms is None or val_transforms is None or test_transforms is None):
            raise ValueError("transforms is not specified you must specify transforms for train, val and test")

        self.train_kwargs = train_kwargs or {}
        self.val_kwargs = val_kwargs or {}
        self.test_kwargs = test_kwargs or {}

        if train_frontal_only is None:
            train_frontal_only = frontal_only

        if val_frontal_only is None:
            val_frontal_only = frontal_only

        if test_frontal_only is None:
            test_frontal_only = frontal_only

        self.train_kwargs['frontal_only'] = train_frontal_only
        self.val_kwargs['frontal_only'] = val_frontal_only
        self.test_kwargs['frontal_only'] = test_frontal_only

        self.train_transforms = train_transforms or transforms
        self.val_transforms = val_transforms or transforms
        self.test_transforms = test_transforms or transforms

        self.data_folder = data_folder
        self.batch_size = batch_size

        if num_workers < 0:
            num_workers = os.cpu_count()
        self.num_workers = num_workers
        self.output_dir = output_dir

        self.dataset_info = {
            'train_kwargs': self.train_kwargs,
            'val_kwargs': self.val_kwargs,
            'test_kwargs': self.test_kwargs,
            'batch_size': self.batch_size,
            "num_workers": self.num_workers,
            "output_dir": self.output_dir,
        }

    @property
    def labels(self):
        raise NotImplementedError()

    def load_datasets(self, stage=None) -> None:
        pass

    def setup(self, stage=None) -> None:
        self.load_datasets(stage=stage)
        self.dataset_info["dataset_len"] = {
            "train": len(self.train_dataset),
            "val": len(self.val_dataset),
            "test": len(self.test_dataset),
        }
        self.dataset_info['label'] = self.labels
        if self.trainer.is_global_zero:
            self.save_info()

    def save_info(self):
        output_dir = os.path.join(self.output_dir, "data")
        print(f"Saving data info to {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, "info.yml"), 'w') as f:
            yaml.safe_dump(self.dataset_info, f)

        self.save_datasets(output_dir)

        return output_dir

    def save_datasets(self, output_dir):
        self.train.to_csv(os.path.join(output_dir, "train.csv"))
        self.val.to_csv(os.path.join(output_dir, "valid.csv"))
        self.test.to_csv(os.path.join(output_dir, "test.csv"))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if not len(self.test_dataset):
            return None
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    @classmethod
    def add_argparse_args(cls, parser, **kwargs):
        group = parser.add_argument_group("data")
        group.add_argument(
            "--data_folder", type=str, dest="data_folder", help="data folder", required=True, )
        group.add_argument("--batch_size", type=int, dest="batch_size", help="batch_size", default=64)
        group.add_argument("--num_workers", type=int, dest="num_workers", help="number of workers for loading",
                           default=-1, )
        group.add_argument("--frontal_only", type=int, dest="frontal_only", help="",
                           default=0, )
        group.add_argument("--train_frontal_only", type=int, dest="train_frontal_only", help="",
                           default=None, )
        group.add_argument("--val_frontal_only", type=int, dest="val_frontal_only", help="",
                           default=None, )
        group.add_argument("--test_frontal_only", type=int, dest="test_frontal_only", help="",
                           default=None, )
        return parser


class CheXpertDataModule(BaseDatamodule):

    def load_datasets(self, stage=None) -> None:
        self.train = pd.read_csv(os.path.join(self.data_folder, 'train.csv'), low_memory=False)

        self.train_dataset = ChestXrayDataset(
            self.data_folder,
            self.train,
            transform=self.train_transforms,
        )

        self.val = pd.read_csv(os.path.join(self.data_folder, 'valid.csv'), low_memory=False)
        self.val_dataset = ChestXrayDataset(
            self.data_folder,
            self.val,
            transform=self.val_transforms,
        )

        self.test = pd.DataFrame([])
        self.test_dataset = []
        self.predict_dataset = []


class PediatricCheXpertDataModule(BaseDatamodule):

    def __init__(self,
                 data_folder,
                 transforms=None,
                 train_transforms=None,
                 test_transforms=None,
                 batch_size=32,
                 num_workers=-1,
                 train_kwargs=None,
                 test_kwargs=None,
                 output_dir='./',
                 ):
        super().__init__(data_folder,
                         transforms=transforms,
                         train_transforms=train_transforms,
                         test_transforms=test_transforms,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         train_kwargs=train_kwargs,
                         test_kwargs=test_kwargs,
                         output_dir=output_dir, )

    def load_datasets(self, stage=None) -> None:
        self.train = pd.read_csv(os.path.join(self.data_folder, 'train_image_data.csv'), low_memory=False)

        self.train_dataset = PediatricChestXrayDataset(
            self.data_folder,
            self.train,
            transform=self.train_transforms,
        )

        # No validation set
        self.val = pd.read_csv(os.path.join(self.data_folder, 'test_image_data.csv'), low_memory=False)
        self.val_dataset = PediatricChestXrayDataset(
            self.data_folder,
            self.val,
            transform=self.test_transforms,
        )

        self.test = pd.read_csv(os.path.join(self.data_folder, 'test_image_data.csv'), low_memory=False)
        self.test_dataset = PediatricChestXrayDataset(
            self.data_folder,
            self.test,
            transform=self.test_transforms,
        )

        self.predict_dataset = PediatricChestXrayDataset(
            self.data_folder,
            self.train,
            transform=self.train_transforms,
        )

    @property
    def labels(self):
        # Proxy, not needed for VAE
        return list(LABEL_MAP)

    def predict_dataloader(self):
        if not len(self.predict_dataset):
            return None
        return DataLoader(
            self.predict_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @classmethod
    def add_argparse_args(cls, parser, **kwargs):
        parser = super().add_argparse_args(parser, **kwargs)
        return parser


class PadChestDataModule(BaseDatamodule):

    def __init__(self,
                 data_folder,
                 label_map_yaml=None,
                 bad_files_yaml=None,
                 csv_file=None,
                 split_dates=settings.PADCHEST_SPLIT_DATES,

                 transforms=None,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None,

                 batch_size=32,
                 num_workers=-1,
                 train_kwargs=None,
                 val_kwargs=None,
                 test_kwargs=None,

                 output_dir='./',

                 frontal_only=False,
                 train_frontal_only=None,
                 val_frontal_only=None,
                 test_frontal_only=None,
                 save_info=True,
                 ):
        super().__init__(data_folder,
                         transforms=transforms,
                         train_transforms=train_transforms,
                         val_transforms=val_transforms,
                         test_transforms=test_transforms,

                         batch_size=batch_size,
                         num_workers=num_workers,
                         train_kwargs=train_kwargs,
                         val_kwargs=val_kwargs,
                         test_kwargs=test_kwargs,

                         output_dir=output_dir,

                         frontal_only=frontal_only,
                         train_frontal_only=train_frontal_only,
                         val_frontal_only=val_frontal_only,
                         test_frontal_only=test_frontal_only, )

        self.csv_file = csv_file or os.path.join(self.data_folder,
                                                 "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
        self.split_dates = split_dates

        if label_map_yaml is not None:
            with open(label_map_yaml, "r") as f:
                label_map = yaml.safe_load(f)
        else:
            label_map = LABEL_MAP

        if bad_files_yaml is not None:
            with open(bad_files_yaml, "r") as f:
                bad_files = yaml.safe_load(f)
        else:
            bad_files = BAD_FILES

        self.label_map = label_map
        self.bad_files = bad_files

        self.dataset_info["split_dates"] = self.split_dates

    @property
    def labels(self):
        return list(self.label_map)

    def save_datasets(self, output_dir):
        super().save_datasets(output_dir)
        self.parent.to_csv(os.path.join(output_dir, "all.csv"))

    def load_datasets(self, stage=None) -> None:
        self.parent = PadChest(self.csv_file, label_map=self.label_map, bad_files=self.bad_files)
        self.train, self.val, self.test = self.parent.split(self.split_dates)

        self.train_dataset = self.train.to_dataset(self.data_folder, labels=self.labels,
                                                   transform=self.train_transforms,
                                                   **self.train_kwargs)

        self.val_dataset = self.val.to_dataset(self.data_folder, labels=self.labels, transform=self.val_transforms,
                                               **self.val_kwargs)

        self.test_dataset = self.test.to_dataset(self.data_folder, labels=self.labels, transform=self.test_transforms,
                                                 **self.test_kwargs)

        self.predict_dataset = self.parent.to_dataset(self.data_folder, labels=self.labels,
                                                      transform=self.test_transforms, **self.test_kwargs)

    def predict_dataloader(self):
        if not len(self.predict_dataset):
            return None
        return DataLoader(
            self.predict_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def save_info(self):
        output_dir = super().save_info()
        with open(os.path.join(output_dir, "label_map.yml"), 'w') as f:
            yaml.safe_dump(self.label_map, f)

        with open(os.path.join(output_dir, "bad_files.yml"), 'w') as f:
            yaml.safe_dump(self.bad_files, f)

    @classmethod
    def add_argparse_args(cls, parser, **kwargs):
        parser = super().add_argparse_args(parser, **kwargs)
        group = parser.add_argument_group("padchest")

        group.add_argument('--split_dates', help="split dates", dest="split_dates", type=_split_dates, nargs=2,
                           default=settings.PADCHEST_SPLIT_DATES_STR)
        group.add_argument(
            "--label_map_yaml", type=str, dest="label_map_yaml", help="yaml file with a new label mapping",
            default=None)
        group.add_argument(
            "--bad_files_yaml", type=str, dest="bad_files_yaml", help="yaml file with bad files", default=None)

        return parser


class MIDRCDataModule(BaseDatamodule):
    __dataset_cls__ = MIDRCDataset

    def __init__(self,
                 data_folder,
                 transforms=None,
                 train_transforms=None,
                 test_transforms=None,
                 batch_size=32,
                 num_workers=-1,
                 train_kwargs=None,
                 test_kwargs=None,
                 output_dir='./',
                 ):
        super().__init__(data_folder,
                         transforms=transforms,
                         train_transforms=train_transforms,
                         test_transforms=test_transforms,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         train_kwargs=train_kwargs,
                         test_kwargs=test_kwargs,
                         output_dir=output_dir, )

    def load_datasets(self, stage=None) -> None:
        dataframe = pd.read_csv(os.path.join(self.data_folder, ), index_col=0)

        # all one dataset for now
        self.train = self.val = self.test = dataframe

        self.train_dataset = self.__dataset_cls__(
            self.data_folder,
            self.train,
            transform=self.train_transforms,
            labels=self.labels
        )

        self.val_dataset = self.__dataset_cls__(
            self.data_folder,
            self.val,
            transform=self.test_transforms,
            labels=self.labels
        )
        self.test_dataset = self.__dataset_cls__(
            self.data_folder,
            self.test,
            transform=self.test_transforms,
            labels=self.labels
        )

        self.predict_dataset = self.__dataset_cls__(
            self.data_folder,
            self.train,
            transform=self.train_transforms,
            labels=self.labels
        )

    @property
    def labels(self):
        return list(LABEL_MAP)

    def predict_dataloader(self):
        if not len(self.predict_dataset):
            return None
        return DataLoader(
            self.predict_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class MGBCXRDataModule(BaseDatamodule):
    __dataset_cls__ = MGBCXRDataset

    ALLOWABLE_SOP_CLASSES = (
        pydicom.uid.ComputedRadiographyImageStorage,
        pydicom.uid.DigitalXRayImageStorageForPresentation,
    )
    ALLOWABLE_MODALITIES = ('CR', 'DX')
    ALLOWABLE_BODY_PARTS = ('CHEST',)
    ALLOWABLE_PIS = ('MONOCHROME1', 'MONOCHROME2')
    
    

    def __init__(
        self,
        data_folder,
        csv_folder=mgb_locations.csv_dir,
        labels_csv=mgb_locations.preprocessed_labels_csv,
        transforms=None,

        train_transforms=None,
        val_transforms=None,
        test_transforms=None,

        batch_size=32,
        num_workers=-1,
        train_kwargs=None,
        val_kwargs=None,
        test_kwargs=None,

        output_dir='./',

        frontal_only=False,
        train_frontal_only=None,
        val_frontal_only=None,
        test_frontal_only=None,
        cache_folder=None,
    ):
        super().__init__(
            data_folder=data_folder,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            train_kwargs=train_kwargs,
            val_kwargs=val_kwargs,
            test_kwargs=test_kwargs,
            output_dir=output_dir,
            frontal_only=frontal_only,
            train_frontal_only=train_frontal_only,
            val_frontal_only=val_frontal_only,
            test_frontal_only=test_frontal_only,
        )
        self.csv_folder = Path(csv_folder)
        self.labels_csv = Path(labels_csv)
        self.cache_folder = Path(cache_folder) if cache_folder is not None else None

    @property
    def labels(self):
        return self.__dataset_cls__.LABEL_COLUMNS

    def load_datasets(self, stage=None) -> None:
        labels_df = pd.read_csv(
            self.labels_csv,
            dtype={
                'AccessionNumber': str,
                'PatientID': str,
                'StudyInstanceUID': str,
            },
            index_col=0,
        )
        labels_df = labels_df[labels_df.StudyDate.notnull()].copy()
        labels_df['StudyDate'] = labels_df.StudyDate.apply(
            lambda x: datetime.datetime.strptime(x, '%m/%d/%Y')
        )
        train_labels_df = labels_df[
            labels_df.StudyDate < mgb_data.TRAIN_DATE_END
        ].copy()
        val_labels_df = labels_df[
            (labels_df.StudyDate > mgb_data.TRAIN_DATE_END) &
            (labels_df.StudyDate < mgb_data.VAL_DATE_END)
        ].copy()

        dcm_df = pd.read_csv(
            self.csv_folder / "dicom_inventory.csv",
            dtype=str,
            index_col=0,
        )

        # Strip 0s from IDs so that they match between dataframes
        dcm_df['PatientID'] = dcm_df.PatientID.str.lstrip('0')
        dcm_df['AccessionNumber'] = dcm_df.AccessionNumber.str.lstrip('0')

        # Apply basic inclusion/exclusion criteria
        dcm_df["is_frontal"] = dcm_df.ViewPosition.isin(('AP', 'PA'))
        dcm_df = dcm_df[
            dcm_df.SOPClassUID.isin(self.ALLOWABLE_SOP_CLASSES) &
            dcm_df.Modality.isin(self.ALLOWABLE_MODALITIES) &
            dcm_df.BodyPartExamined.isin(self.ALLOWABLE_BODY_PARTS) &
            dcm_df.PhotometricInterpretation.isin(self.ALLOWABLE_PIS)
        ].copy()

        self.train = train_labels_df.merge(
            dcm_df,
            how='inner',
            on=('PatientID', 'AccessionNumber', 'StudyInstanceUID'),
        )
        if self.train_kwargs["frontal_only"]:
            self.train = self.train[self.train.is_frontal].copy()
            
        

        # Create train_dataset instance and set dataset_type attribute
        self.train_dataset = self.__dataset_cls__(
            self.data_folder,
            self.train,
            transform=self.train_transforms,
            **self.train_kwargs,
        )
        
        #if self.cache_folder is None:
        #    print(f"There are {len(self.train_dataset.image_paths)} images in the train dataset")
        #    self.train_dataset.dataset_type = 'train'
        #    self.__dataset_cls__.load_data_into_memory(self.data_folder, self.train_dataset.image_paths, 'train', num_workers=self.num_workers)
            

        self.val = val_labels_df.merge(
            dcm_df,
            how='inner',
            on=('PatientID', 'AccessionNumber', 'StudyInstanceUID'),
        )
        if self.val_kwargs["frontal_only"]:
            self.val = self.val[self.val.is_frontal].copy()

        
        
        self.val_dataset = self.__dataset_cls__(
            self.data_folder,
            self.val,
            transform=self.val_transforms,
            **self.val_kwargs,
        )
        #if self.cache_folder is None:
        #    print(f"There are {len(self.val_dataset.image_paths)} images in the validation dataset")
        #    self.val_dataset.dataset_type = 'val'
        #    self.__dataset_cls__.load_data_into_memory(self.data_folder, self.val_dataset.image_paths, 'val', num_workers=self.num_workers)
        
        # For now, test is simply the entire dataset
        self.test = labels_df.merge(
            dcm_df,
            how='inner',
            on=('PatientID', 'AccessionNumber', 'StudyInstanceUID'),
        )
        self.test_dataset = self.__dataset_cls__(
            self.data_folder,
            self.test,
            transform=self.test_transforms,
            **self.test_kwargs,
        )

        if self.cache_folder is not None:
            print(f"Creating cache at {self.cache_folder}")
            self.train_dataset.ensure_cache(self.cache_folder, self.num_workers)
            self.val_dataset.ensure_cache(self.cache_folder, self.num_workers)
            self.test_dataset.ensure_cache(self.cache_folder, self.num_workers)
            print(f"Done creating cache.")

    @classmethod
    def add_argparse_args(cls, parser, **kwargs):
        parser = super().add_argparse_args(parser, **kwargs)
        group = parser.add_argument_group("mgb")

        group.add_argument(
            "--csv_folder",
            type=Path,
            help="Path to the CSV directory",
            default=mgb_locations.csv_dir,
        )
        group.add_argument(
            "--labels_csv",
            type=Path,
            help="Path to the CSV file containing labels",
            default=mgb_locations.preprocessed_labels_csv,
        )
        group.add_argument(
            "--cache_folder",
            type=Path,
            help="Use this location to cache decompressed data.",
        )

        return parser
