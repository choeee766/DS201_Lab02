from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def build_transforms(img_size, augment=True):
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf

def make_loaders(data_root, img_size, augment=True, batch_size=32,
                 val_ratio=0.10, num_workers=2, seed=42):
    root = Path(data_root)
    train_tf, eval_tf = build_transforms(img_size, augment)
    full_train = datasets.ImageFolder(root / "train", transform=train_tf)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)

    if (root / "val").is_dir():
        train_ds = full_train
        val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
        val_classes = val_ds.class_to_idx
    else:
        n_val = int(len(full_train) * val_ratio)
        if not 0 < n_val < len(full_train):
            raise ValueError("val_ratio tạo ra tập train hoặc validation rỗng")
        train_part, val_part = random_split(
            full_train, [len(full_train) - n_val, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        val_images = datasets.ImageFolder(root / "train", transform=eval_tf)
        train_ds = train_part
        val_ds = Subset(val_images, val_part.indices)
        val_classes = val_images.class_to_idx

    if len(full_train.classes) != 21:
        raise ValueError(f"VinaFood21 cần 21 lớp, đang có {len(full_train.classes)}")
    if full_train.class_to_idx != val_classes:
        raise ValueError("Nhãn lớp train/validation không khớp")
    if full_train.class_to_idx != test_ds.class_to_idx:
        raise ValueError("Nhãn lớp train/test không khớp")

    common = dict(batch_size=batch_size, num_workers=num_workers)
    train_dl = DataLoader(train_ds, shuffle=True,
                          generator=torch.Generator().manual_seed(seed), **common)
    val_dl = DataLoader(val_ds, shuffle=False, **common)
    test_dl = DataLoader(test_ds, shuffle=False, **common)
    return train_dl, val_dl, test_dl, full_train.classes

def make_test_loader(data_root, img_size, batch_size=32, num_workers=2):
    root = Path(data_root)
    _, eval_tf = build_transforms(img_size, augment=False)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers), test_ds.classes
