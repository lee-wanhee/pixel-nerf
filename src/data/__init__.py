import os

from .MultiObjectDataset import MultiObjectDataset
from .DVRDataset import DVRDataset
from .SRNDataset import SRNDataset
from .TDWDataset import MultiscenesDataset
from .MSNDataset import Clevr3dDataset

from .data_util import ColorJitterDataset


def get_split_dataset(dataset_type, datadir, want_split="all", training=True, opt=None, **kwargs):
    """
    Retrieved desired dataset class
    :param dataset_type dataset type name (srn|dvr|dvr_gen, etc)
    :param datadir root directory name for the dataset. For SRN/multi_obj data:
    if data is in dir/cars_train, dir/cars_test, ... then put dir/cars
    :param want_split root directory name for the dataset
    :param training set to False in eval scripts
    """
    dset_class, train_aug = None, None
    flags, train_aug_flags = {}, {}

    if dataset_type == "srn":
        # For ShapeNet single-category (from SRN)
        dset_class = SRNDataset
    elif dataset_type == "multi_obj":
        # For multiple-object
        dset_class = MultiObjectDataset
    elif dataset_type.startswith("dvr"):
        # For ShapeNet 64x64
        dset_class = DVRDataset
        if dataset_type == "dvr_gen":
            # For generalization training (train some categories, eval on others)
            flags["list_prefix"] = "gen_"
        elif dataset_type == "dvr_dtu":
            # DTU dataset
            flags["list_prefix"] = "new_"
            if training:
                flags["max_imgs"] = 49
            flags["sub_format"] = "dtu"
            flags["scale_focal"] = False
            flags["z_near"] = 0.1
            flags["z_far"] = 5.0
            # Apply color jitter during train
            train_aug = ColorJitterDataset
            train_aug_flags = {"extra_inherit_attrs": ["sub_format"]}
    elif dataset_type.startswith("tdw"):
        dset_class = MultiscenesDataset
    elif dataset_type.startswith("msn"):
        dset_class = Clevr3dDataset

    else:
        raise NotImplementedError("Unsupported dataset type", dataset_type)

    want_train = want_split != "val" and want_split != "test"
    want_val = want_split != "train" and want_split != "test"
    want_test = want_split != "train" and want_split != "val"

    print(want_train, 'want_train')
    print(want_val, 'want_val')
    print(want_test, 'want_test')
    # breakpoint()
    if dataset_type.startswith("tdw"):
        if want_train:
            opt.isTrain = True
            train_set = dset_class(opt, **flags, **kwargs)

        if want_val:
            opt.isTrain = False
            print(datadir)
            if 'tdw' in datadir:
                opt.val_n_scenes = 600
                opt.n_scenes = 600
            elif 'bridge' in datadir:
                opt.val_n_scenes = 50
                opt.n_scenes = 50
                print(f"'bridge' in {datadir}")
            else:
                raise NotImplementedError
            if 'train' in datadir:
                opt.datadir = datadir.replace('train', 'val')
            print('want_val', opt.datadir)
            val_set = dset_class(opt, **flags, **kwargs)

        if want_test:
            # raise NotImplementedError('test is not implemented yet')
            # TODO: check that this is not being used
            opt.isTrain = False
            opt.skip = 0
            if 'tdw' in datadir:
                print(f'tdw in {datadir}')
                opt.n_scenes = 600
                opt.val_n_scenes = 600
            elif 'bridge' in datadir:
                print(f'bridge in {datadir}')
                opt.n_scenes = 50
                opt.val_n_scenes = 50
            else:
                raise NotImplementedError
            if 'train' in datadir:
                opt.datadir = datadir.replace('train', 'test')
            print('want_test', opt.datadir)
            test_set = dset_class(opt, **flags, **kwargs)

        if want_split == "train":
            return train_set
        elif want_split == "val":
            return val_set
        elif want_split == "test":
            return test_set
        return train_set, val_set, test_set

    elif dataset_type.startswith("msn"):


        if want_train:
            train_set = Clevr3dDataset(path='/ccn2/u/honglinc/datasets/multishapenet', mode='train', max_n=6, max_views=None,
                           points_per_item=2048, do_frustum_culling=False, shapenet=True, max_len=None,
                           importance_cutoff=0.5)
        if want_val:
            val_set = Clevr3dDataset(path='/ccn2/u/honglinc/datasets/multishapenet', mode='val', max_n=6, max_views=None,
                           points_per_item=2048, do_frustum_culling=False, shapenet=True, max_len=None,
                           importance_cutoff=0.5)
        if want_test:
            print('opt.msn_test_mode:', opt.msn_test_mode)
            test_set = Clevr3dDataset(path='/ccn2/u/honglinc/datasets/multishapenet', mode=opt.msn_test_mode, max_n=6,
                                     max_views=None,
                                     points_per_item=2048, do_frustum_culling=False, shapenet=True, max_len=None,
                                     importance_cutoff=0.5)

        if want_split == "train":
            return train_set
        elif want_split == "val":
            return val_set
        elif want_split == "test":
            return test_set
        return train_set, val_set, test_set

    else:
        if want_train:
            train_set = dset_class(datadir, stage="train", **flags, **kwargs)
            if train_aug is not None:
                train_set = train_aug(train_set, **train_aug_flags)

        if want_val:
            val_set = dset_class(datadir, stage="val", **flags, **kwargs)

        if want_test:
            test_set = dset_class(datadir, stage="test", **flags, **kwargs)

        if want_split == "train":
            return train_set
        elif want_split == "val":
            return val_set
        elif want_split == "test":
            return test_set
        return train_set, val_set, test_set
