import argparse
from model_drift.data.datamodules import MGBCXRDataModule
from pycrumbs import tracked


@tracked(directory_parameter="cache_folder")
def make_cache(args, cache_folder):
    # Creating the cache is handled within the prepare_dataset, called by the
    # constructor
    dm = MGBCXRDataModule.from_argparse_args(args, transforms=lambda x: x)
    dm.load_datasets()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Make a cached version of the dataset.")
    parser = MGBCXRDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    if not hasattr(args, "cache_folder"):
        raise TypeError("Must specify a cache dir.")
    make_cache(args, args.cache_folder)
