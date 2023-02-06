import torch
from torch import tensor
from torchvision.models.resnet import resnet34
from PIL import Image
from itertools import compress

import pandas as pd
from pathlib import Path
from fastcore.xtras import Path  # @patched Pathlib.path  # noqa: F401,F811

from fastai.data.core import Datasets, DataLoaders
from fastai.data.external import URLs, untar_data
from fastai.data.transforms import (
    ColReader,
    ToTensor,
    IntToFloatTensor,
    MultiCategorize,
    Normalize,
    OneHotEncode,
    RandomSplitter,
)

from fastai.metrics import accuracy_multi

from torchvision.transforms import PILToTensor
from fastai.vision.augment import aug_transforms, imagenet_stats
from fastai.vision.core import PILImage
from fastai.vision.learner import vision_learner
from fastai.learner import Learner
from fastai.callback.schedule import (  # noqa: F811
    Learner,
)  # @patched Learner functions like lr_find and fit_one_cycle


# # def get_x(row: pd.Series) -> Path:
# #     return (src / "train" / row.image_name).with_suffix(".jpg")


# # def get_y(row: pd.Series) -> List[str]:
# #     return row.tags.split(" ")


def get_datasets() -> Datasets:
    src = untar_data(URLs.PLANET_SAMPLE)
    df = pd.read_csv(src / "labels.csv")

    # get labels
    all_tags = df["tags"].values
    all_labels = []
    for row in all_tags:
        all_labels += row.split(" ")
    different_labels = set(all_labels)

    # filter labels with low appearing
    counts = {label: all_labels.count(label) for label in different_labels}
    for key, count in counts.items():
        if count < 10:
            df = df[df["tags"].str.contains(key) == False]  # noqa: E712

    # items
    get_x = ColReader(0, pref=f"{src}/train/", suff=".jpg")
    get_y = ColReader(1, label_delim=" ")

    # items transformations
    tfms = [
        [get_x, PILImage.create],
        [
            get_y,
            MultiCategorize(vocab=sorted(different_labels)),
            OneHotEncode(len(different_labels)),
        ],
    ]

    # splitter
    train_idxs, valid_idxs = RandomSplitter(valid_pct=0.2, seed=42)(df)

    return Datasets(df, tfms=tfms, splits=[train_idxs, valid_idxs])


def get_dataloaders(bs: int = 64) -> DataLoaders:
    dsets = get_datasets()

    item_tfms = [ToTensor]
    batch_tfms = [
        IntToFloatTensor(),
        *aug_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.0),
        Normalize.from_stats(*imagenet_stats),
    ]

    return dsets.dataloaders(after_item=item_tfms, after_batch=batch_tfms, bs=bs)


def get_learner(bs: int = 64) -> Learner:
    dls = get_dataloaders(bs=bs)
    return vision_learner(dls, resnet34, metrics=[accuracy_multi])


def predict(fname: str, learn: Learner = None):
    im = Image.open(fname)
    im = im.convert("RGB")
    t_im = PILToTensor()(im)

    t_im = t_im.unsqueeze(0)  # batch of one
    t_im = t_im.float().div_(255.0)  # Float to tensor

    # apply ImageNet normalization
    mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    vector = [1] * 4
    vector[1] = -1
    mean = tensor(mean).view(*vector)  # to shape torch.Size([1, 3, 1, 1])
    std = tensor(std).view(*vector)  # to shape torch.Size([1, 3, 1, 1])
    t_im = (t_im - mean) / std

    # model
    if learn is None:
        learn = get_learner()
        learn = learn.load("weights")
        learn.cuda()

    # vocab
    labels = learn.dls.vocab

    # predict
    with torch.inference_mode():
        learn.model.eval()
        preds = learn.model(t_im.cuda())

    decoded_preds = torch.sigmoid(preds) > 0.5

    # The compress function creates an iterator that filters elements based
    # on some boolean array, which is what our decoded_preds are originally
    present_labels = list(compress(data=labels, selectors=decoded_preds[0]))

    return present_labels
