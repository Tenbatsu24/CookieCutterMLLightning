import wandb
import numpy as np
import torchvision.transforms as T

from loguru import logger
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from ml.util import STORE, DATA_TYPE, AUG_TYPE
from ml.config import PROCESSED_DATA_DIR, NUM_WORKERS


class LabelNoiseTransform:
    def __init__(self, noise, num_classes, seed):
        self.noise = noise
        self.num_classes = num_classes
        self.random_state = np.random.RandomState(seed)

    def __call__(self, target):
        if self.random_state.rand() < self.noise:
            # Randomly select a different class
            target = self.random_state.choice(list(set(range(self.num_classes)) - {target}), 1)[0]
        return target


def handle_subset(cfg, ds_name, train_dataset):
    if hasattr(cfg, "subset") and 0 < cfg.subset.pct < 100:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=cfg.subset.pct / 100, random_state=cfg.subset.subset_seed
        )
        indices = list(range(len(train_dataset)))
        labels = (
            train_dataset.targets
            if ds_name != "im10"
            else [label for _, label in train_dataset._samples]
        )
        for _, test_indices in splitter.split(indices, labels):
            indices = test_indices
        train_dataset = Subset(train_dataset, indices)
    return train_dataset


def handle_label_noise(cfg, ds_name, train_dataset):
    if hasattr(cfg, "label_noise") and cfg.label_noise.rate > 0:
        label_noise_transform = LabelNoiseTransform(
            cfg.label_noise.rate, cfg.num_classes, cfg.label_noise.noise_seed
        )
        if ds_name == "im10":
            noised_samples = [
                (path, label_noise_transform(label)) for path, label in train_dataset._samples
            ]
            train_dataset._samples = noised_samples
        else:
            train_dataset.targets = [
                label_noise_transform(label) for label in train_dataset.targets
            ]


def get_standard_transforms(cfg):
    ds_name = cfg.dataset.name

    val_transform = []

    if hasattr(cfg, "finetune") and cfg.finetune.enable:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        crop, resize = cfg.finetune.crop, cfg.finetune.resize

        train_transform = [
            T.RandomResizedCrop(crop, antialias=True),
            T.RandomHorizontalFlip(),
        ]
        val_transform = [
            T.Resize(resize),
            T.CenterCrop(crop),
        ]
    elif ds_name in ["c10", "c100"]:
        mean, std = (0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)
        train_transform = [
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
        ]
    elif ds_name in ["m", "fm", "km"]:
        mean, std = (0.1307,), (0.3081,)
        train_transform = [
            T.RandomCrop(28, padding=4),
        ]
    elif ds_name in ["im10", "in100", "in"]:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        crop, resize = 224, 256

        train_transform = [
            T.RandomResizedCrop(crop, antialias=True),
            T.RandomHorizontalFlip(),
        ]
        val_transform = [
            T.Resize(resize),
            T.CenterCrop(crop),
        ]
    else:
        raise ValueError(f"Unknown dataset {ds_name}")

    if hasattr(cfg.dataset, "aug"):
        for aug in cfg.dataset.aug:
            aug = ConfigDict(aug, convert_dict=True)
            try:
                aug_cls = STORE.get(AUG_TYPE, aug.type)
                train_transform.append(aug_cls(**aug.params))
            except KeyError:
                if hasattr(T, aug.type):
                    train_transform.append(getattr(T, aug.type)(**aug.params))
                else:
                    raise ValueError(f"Unknown augmentation {aug.type}")

    train_transform = T.Compose(train_transform + [T.ToTensor()])
    val_transform = T.Compose(val_transform + [T.ToTensor()])

    return train_transform, val_transform, mean, std


def make_loaders(cfg):
    ds_name = cfg.dataset.name

    ds = STORE.get(DATA_TYPE, ds_name)

    train_transform, val_transform, mean, std = get_standard_transforms(cfg)

    if ds_name == "im10":
        train_dataset = ds(PROCESSED_DATA_DIR, split="train", transform=train_transform)
    else:
        train_dataset = ds(
            PROCESSED_DATA_DIR, train=True, download=True, transform=train_transform
        )

    handle_label_noise(cfg, ds_name, train_dataset)

    train_dataset = handle_subset(cfg, ds_name, train_dataset)

    if ds_name == "im10":
        val_dataset = ds(PROCESSED_DATA_DIR, split="val", transform=val_transform)
    else:
        val_dataset = ds(PROCESSED_DATA_DIR, train=False, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
    )

    return train_loader, val_loader, T.Normalize(mean=mean, std=std), val_transform


def generalisation_test(cfg, module, test_transform):
    if not hasattr(cfg.dataset, "gen_test"):
        return

    gen_test_ds_s = cfg.dataset.gen_test

    for gen_ds in gen_test_ds_s:
        gen_ds = ConfigDict(gen_ds, convert_dict=True)
        c_dataset_class = STORE.get(DATA_TYPE, gen_ds.name)

        if gen_ds.type == "corruption":

            metric_keys = [k for k in module.test_metrics.keys()]

            averages = [0 for _ in metric_keys]
            counts = 0

            my_table = wandb.Table(columns=["corruption", "severity", *metric_keys])
            for corruption in c_dataset_class.corruptions:
                for severity in c_dataset_class.severities:
                    c_s_dst = c_dataset_class(
                        PROCESSED_DATA_DIR,
                        download=False,
                        transform=test_transform,
                        severity=severity,
                        corruption=corruption,
                    )

                    c_s_loader = DataLoader(
                        c_s_dst,
                        batch_size=cfg.batch_size,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                    )

                    c_s_log = module.manual_test(
                        c_s_loader, name=f"{corruption}_{severity}".ljust(50)
                    )
                    metrics = [c_s_log[k] for k in metric_keys]
                    entry = [corruption, severity, *metrics]

                    my_table.add_data(*entry)
                    for k_idx, _ in enumerate(metric_keys):
                        averages[k_idx] += metrics[k_idx]
                    counts += 1

            # take the mean of the metrics and add a summary entry which is the mean
            averages = [v / counts for v in averages]
            average_row = ["average", 3, *averages]
            my_table.add_data(*average_row)

            wandb.log({f"{gen_ds.name}": my_table})

        else:
            if gen_ds.name == "stl":
                stl_transform = T.Compose([T.Resize((32, 32)), test_transform])
            else:
                logger.error(f"Unknown gen_ds: {gen_ds}")
                return

            gen_dst = c_dataset_class(
                PROCESSED_DATA_DIR, split="test", download=False, transform=stl_transform
            )
            gen_loader = DataLoader(
                gen_dst,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            gen_log = module.manual_test(gen_loader, name=f"{gen_ds.name}".ljust(50))

            # add prefix gen_ds
            gen_log = {f"{gen_ds.name}/{k}": v for k, v in gen_log.items()}
            wandb.log(gen_log)
