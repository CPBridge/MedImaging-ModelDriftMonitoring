#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import multiprocessing as mp
import os
from pathlib import Path
from time import time
from typing import Union


import numpy as np
import pandas as pd
import six
import torch
from PIL import Image
from PIL import ImageFile
import pydicom
from torch.utils.data import Dataset

from model_drift.data import mgb_data


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _trunc_long_str(s, max_size):
    s = str(s)
    if len(s) <= max_size:
        return s
    n_2 = int(max_size / 2 - 3)
    n_1 = max_size - n_2 - 3
    return "{0}...{1}".format(s[:n_1], s[-n_2:])


def normalize_PIL(image):
    if image.mode in ["I"]:
        image = Image.fromarray((np.array(image) / 256).astype(np.uint8))
    return image.convert("RGB")


class BaseDataset(Dataset):
    def __init__(
            self,
            folder_dir,
            dataframe_or_csv,
            transform=None,
            frontal_only=False,
            image_dir=None,
            labels=None,
    ):
        """
        Init Dataset

        Parameters
        ----------
        folder_dir: str
            folder contains all images
        dataframe_path: CSV
            dataframe_path csv contains all information of images

        image_size: int
            image size to rescale
        normalization: bool
            whether applying normalization with mean and std from ImageNet or not
        """

        self.dataframe_or_csv = dataframe_or_csv
        self.folder_dir = folder_dir
        self.image_dir = image_dir
        self.frontal_only = frontal_only
        self.labels = labels
        self.image_transformation = transform
        self._reset_lists()
        self.prepare_data()

    def read_csv(self, csv):
        return pd.read_csv(csv, low_memory=False)

    def __str__(self) -> str:

        params = [
            f"folder_dir={_trunc_long_str(self.folder_dir, 30)}",
            f"labels = {self.labels}",
            f"frontal_only={self.frontal_only}",
        ]
        return f"{type(self).__name__}(" + ", ".join(params) + ")"

    def prepare_data(self):
        print("Initializing:", str(self))

    def _reset_lists(self):
        self.image_paths = []  # List of image paths
        self.image_labels = []  # List of image labels
        self.image_index = []
        self.frontal = []
        self.recon_image_path = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        """
        Read image at index and convert to torch Tensor
        """

        # Read image
        image_path = self.image_paths[index]
        image_data_original = self.read_image(image_path)
        # if os.path.exists(image_path):
        # image_data = cv2.cvtColor(cv2.imread(image_path).astype('uint8'), cv2.COLOR_BGR2RGB)
        # Resize and convert image to torch tensor
        image_data = self.image_transformation(image_data_original)
        # label = torch.tensor(self.image_labels[index], dtype=torch.long)
        # Return LOADING TIME

        onp = np.array(image_data_original)

        return {
            "image": image_data,
            "label": torch.FloatTensor(self.image_labels[index]),
            "frontal": torch.FloatTensor([self.frontal[index]]),
            "index": self.image_index[index],
            "recon_path": self.recon_image_path[index],
            "o_mean": torch.tensor([onp.mean()], dtype=torch.float),
            "o_max": torch.tensor([onp.max()], dtype=torch.float),
            "o_min": torch.tensor([onp.min()], dtype=torch.float),
        }

    def read_image(self, image_path):
        try:
            image_data_original = Image.open(image_path)
        except BaseException:
            print(f"\nbad path: {image_path}\n")
            raise
        return normalize_PIL(image_data_original)


class ChestXrayDataset(BaseDataset):
    def prepare_data(self):
        # Get all image paths and image labels from dataframe
        if isinstance(self.dataframe_or_csv, six.string_types):
            print(self.dataframe_or_csv)
            dataframe = pd.read_csv(self.dataframe_or_csv, low_memory=False)
        else:
            dataframe = self.dataframe_or_csv

        dataframe["Frontal"] = dataframe["Frontal/Lateral"] == "Frontal"
        if self.frontal_only:
            dataframe = dataframe[dataframe["Frontal"].astype(bool)]
        for _, row in dataframe.iterrows():
            # Read in image from path
            # print(row)
            image_path = os.path.join(
                self.image_dir or self.folder_dir,
                row.Path.partition("CheXpert-v1.0/")[2],
            )
            self.image_paths.append(image_path)
            # if len(row) < 10:
            labels = [0] * 14
            self.frontal.append(float(row["Frontal"]))
            self.image_labels.append(labels)
            self.image_index.append(row.Path)
            self.recon_image_path.append(row.Path.partition("CheXpert-v1.0/")[2])


class PediatricChestXrayDataset(BaseDataset):
    def prepare_data(self):
        # Get all image paths and image labels from dataframe
        if isinstance(self.dataframe_or_csv, six.string_types):
            # print(self.dataframe_or_csv)
            dataframe = pd.read_csv(self.dataframe_or_csv, low_memory=False)
        else:
            dataframe = self.dataframe_or_csv

        for _, row in dataframe.iterrows():
            # Read in image from path
            image_path = os.path.join(
                self.image_dir or self.folder_dir,
                row.Path.partition("Pediatric_Chest_X-ray_Pneumonia/")[2],
            )
            self.image_paths.append(image_path)

            labels = []
            # Labels come from column after path
            for col in row[1:]:
                if col == 1:
                    labels.append(1)
                else:
                    labels.append(0)
            self.image_labels.append(labels)
            self.frontal.append(1.0)
            self.image_index.append(row.Path)
            self.recon_image_path.append(row.Path.partition("Pediatric_Chest_X-ray_Pneumonia/")[2])


class PadChestDataset(BaseDataset):

    def __init__(self, folder_dir, *args, **kwargs):
        kwargs.setdefault("image_dir", os.path.join(folder_dir, "png"))
        super().__init__(folder_dir, *args, **kwargs)

    def prepare_data(self):
        # Get all image paths and image labels from dataframe
        # Get all image paths and image labels from dataframe
        if isinstance(self.dataframe_or_csv, six.string_types):
            dataframe = pd.read_csv(self.dataframe_or_csv, low_memory=False)
        else:
            dataframe = self.dataframe_or_csv

        if self.labels is not None:
            dataframe["binary_label"] = dataframe[self.labels].apply(list, axis=1)

        if "Frontal" not in dataframe:
            dataframe["Frontal"] = dataframe["Projection"].isin(["PA", "AP", "AP_horizontal"])

        if self.frontal_only:
            dataframe = dataframe[dataframe["Frontal"].astype(bool)]

        for index, row in dataframe.iterrows():
            # Read in image from path
            # print(row)
            image_path = os.path.join(str(int(row["ImageDir"])), str(row["ImageID"]))
            self.image_paths.append(os.path.join(self.image_dir, image_path))
            # if len(row) < 10:
            labels = row["binary_label"] if self.labels is not None else [0]
            self.frontal.append(float(row["Frontal"]))
            self.image_labels.append(labels)
            self.image_index.append(row["ImageID"])
            self.recon_image_path.append(image_path)


class MIDRCDataset(BaseDataset):
    def prepare_data(self):
        # Get all image paths and image labels from dataframe
        if isinstance(self.dataframe_or_csv, six.string_types):
            print(self.dataframe_or_csv)
            dataframe = pd.read_csv(self.dataframe_or_csv, low_memory=False)
        else:
            dataframe = self.dataframe_or_csv

        dataframe["Frontal"] = True
        if self.frontal_only:
            dataframe = dataframe[dataframe["Frontal"].astype(bool)]

        for _, row in dataframe.iterrows():
            # Read in image from path
            image_path = os.path.join(
                self.image_dir or self.folder_dir, 'png',
                row['ImageId'][:-3] + 'png',
            )
            self.image_paths.append(image_path)
            labels = row[self.labels].astype(int).tolist()
            self.frontal.append(float(row["Frontal"]))
            self.image_labels.append(labels)
            self.image_index.append(row['ImageId'][:-3] + 'png')
            self.recon_image_path.append(row['ImageId'][:-3] + 'png')


class MGBCXRDataset(BaseDataset):
    LABEL_COLUMNS = list(mgb_data.LABEL_GROUPINGS.keys())
    data = {}  # Class variable to store data in memory

    def prepare_data(self):
        self.has_cache = False
        if isinstance(self.dataframe_or_csv, pd.DataFrame):
            df = self.dataframe_or_csv
        else:
            df = pd.read_csv(self.dataframe_or_csv, dtype=str, index_col=0)

        self.folder_dir = Path(self.folder_dir)

        for _, row in df.iterrows():
            image_path = (
                self.folder_dir /
                (
                    f"studies/{row.StudyInstanceUID}/series/"
                    f"{row.SeriesInstanceUID}/instances/{row.SOPInstanceUID}"
                )
            )
            self.image_paths.append(image_path)

            labels = []
            for c in self.LABEL_COLUMNS:
                val = float(row[c])
                if val == 1.0 or val == -1.0:
                    # NB equivocal treated as positive
                    labels.append(1)
                else:
                    # NB this means that "not mentioned" is treated as negative
                    labels.append(0)
            self.image_labels.append(labels)

            self.frontal.append(float(row.is_frontal))
            image_id = f"{row.PatientID}_{row.AccessionNumber}_{row.SOPInstanceUID}"
            self.image_index.append(image_id)
            self.recon_image_path.append(image_id + '.png')

    
    def read_image(self, image_path) -> Image:
        """Read an image from the path."""
        if hasattr(self, "has_cache") and self.has_cache:
            # Read in cached version as npy
            arr = np.load(image_path)
            im = torch.tensor(arr)
        else:
            # Read in from raw DICOM
            im = self.read_from_dicom(image_path)
        #im = Image.fromarray(arr)
        #im = im.convert("RGB")
        return im

    #@staticmethod
    #def read_image(image_path) -> Image:
    #    """Read an image from the path."""
    #    
    #    # Read in from raw DICOM
    #    arr = self.read_from_dicom(image_path)
    #    im = Image.fromarray(arr)
    #    im = im.convert("RGB")
    #    return im


    
    #modified to work with reading from memory, 
    def __getitem__(self, index):
        dataset_type = self.dataset_type  # Ensure you set this attribute when creating the dataset instance

        # Check if data is loaded into memory
        if self.data is not None and dataset_type in self.data:
            image_data_original = self.data[dataset_type][index]
            #image_data_original = Image.fromarray(image_data_original_ram)
            #image_data_original = image_data_original.convert("RGB")
        else:
            image_path = self.image_paths[index]
            image_data_original = self.read_image(image_path)

        if len(image_data_original.shape) == 2:
            image_data_original = image_data_original.unsqueeze(0).expand(3, -1, -1)

        
        image_data_original = image_data_original.float()
        image_data = self.image_transformation(image_data_original)
        
        onp = image_data_original

        return {
            "image": image_data,
            "label": torch.FloatTensor(self.image_labels[index]),
            "frontal": torch.FloatTensor([self.frontal[index]]),
            "index": self.image_index[index],
            "recon_path": self.recon_image_path[index],
            # "o_mean": torch.tensor([onp.mean()], dtype=torch.float),
            # "o_max": torch.tensor([onp.max()], dtype=torch.float),
            # "o_min": torch.tensor([onp.min()], dtype=torch.float),
            "o_mean": onp.mean(), 
            "o_max": onp.max(), 
            "o_min": onp.min(), 
        }
        

    @staticmethod
    def read_and_store_image(cls, image_path, dataset_type, data_dict):
        image = cls.read_from_dicom(image_path, return_numpy=True)
        data_dict[dataset_type].append(image)
    
    @classmethod
    def load_data_into_memory(cls, folder_dir, image_paths, dataset_type, num_workers=0):
        start_time = time()
        print(f"Started loading {dataset_type} images into memory")
        with mp.Manager() as manager:
            data_dict = manager.dict()
            data_dict[dataset_type] = manager.list()  
            args = [(cls, path, dataset_type, data_dict) for path in image_paths]
            if num_workers > 0:
                with mp.Pool(num_workers) as p:
                    p.starmap(cls.read_and_store_image, args)
            else:
                for arg in args:
                    cls.read_and_store_image(*arg)
            cls.data = {dataset_type: [torch.tensor(arr, dtype=torch.float).unsqueeze(0).expand(3, -1, -1) for arr in data_dict[dataset_type]]}
        elapsed_time = time() - start_time
        print(f"Finished loading {len(cls.data[dataset_type])} images into memory for {dataset_type} in {elapsed_time:.2f} seconds")


    @staticmethod
    def read_from_dicom(image_path, return_numpy: bool = False) -> Union[np.ndarray, torch.Tensor]:
        dcm = pydicom.dcmread(image_path)
        arr = dcm.pixel_array
        max_val = 2 ** dcm.BitsStored - 1
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            arr = max_val - arr
        arr_min = arr.min()
        arr_max = arr.max()
        arr = ((arr - arr_min) / (arr_max - arr_min)) * 255
        arr = arr.astype(np.uint8)
        if return_numpy: 
            return arr
        else:
            im_tensor = torch.tensor(arr)
            return im_tensor

    @staticmethod
    def _cache_image(dicom_path: Path, numpy_path: Path):
        """Create a cached version of a DICOM file."""
        if not numpy_path.exists():
            arr = MGBCXRDataset.read_from_dicom(dicom_path, return_numpy=True)
            numpy_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(numpy_path, arr)

    def ensure_cache(self, cache_dir: Path, num_workers: int) -> None:
        """Ensures that the given directory contains npy cached versions of
        all images in the datasets."""
        cache_dir.mkdir(exist_ok=True)
        cache_paths = [
            cache_dir / (str(image_path.relative_to(self.folder_dir)) + ".npy")
            for image_path in self.image_paths
        ]
        if not cache_paths[0].exists():
            args = zip(self.image_paths, cache_paths)
            if num_workers > 0:
                with mp.Pool(num_workers) as p:
                    p.starmap(MGBCXRDataset._cache_image, args)
            else:
                for impath, cachepath in args:
                    MGBCXRDataset._cache_image(impath, cachepath)
        self.image_paths = cache_paths
        self.has_cache = True
