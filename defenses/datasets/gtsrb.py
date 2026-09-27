"""German Traffic Sign Recognition Benchmark (GTSRB), 43 classes, cached at 32x32.

Reimplemented on :class:`~defenses.datasets.cached32.Cached32Dataset`. The
previous loader exposed ``samples`` as a list of path *strings*, while the
transfer-set adversary reads ``queryset.samples[i][0]`` expecting ``(path,
label)`` tuples -- so it stored the first character of every path as the "image
path" and substitute training on a GTSRB transfer set could not load any image.
The cached version exposes ``data``/``targets`` like CIFAR10 instead.

Images are resized to 32x32 without cropping the ROI (same as torchvision's
GTSRB). Note that the GTSRB model family uses *no* horizontal-flip augmentation:
mirroring turns e.g. "keep right" (38) into "keep left" (39) without changing
the label.
"""

import csv
from pathlib import Path

from defenses.datasets.cached32 import (
    Cached32Dataset,
    download_file,
    extract_archive,
    images_to_array,
)

GTSRB_BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"


class GTSRB(Cached32Dataset):
    folder = "GTSRB"
    classes = list(range(43))

    @property
    def _base_folder(self) -> Path:
        return self.root / "gtsrb"

    def _training_root(self) -> Path:
        return self._base_folder / "GTSRB" / "Training"

    def _test_root(self) -> Path:
        return self._base_folder / "GTSRB" / "Final_Test" / "Images"

    def download(self) -> None:
        archives = []
        if not self._training_root().is_dir():
            archives.append("GTSRB-Training_fixed.zip")
        if not self._test_root().is_dir():
            archives.append("GTSRB_Final_Test_Images.zip")
        if not (self._base_folder / "GT-final_test.csv").exists():
            archives.append("GTSRB_Final_Test_GT.zip")
        for name in archives:
            local = download_file(GTSRB_BASE_URL + name, self._base_folder / "archives" / name)
            extract_archive(local, self._base_folder)

    def _build(self, split: str):
        if split == "train":
            root = self._training_root()
            if not root.is_dir():
                raise FileNotFoundError(f"GTSRB training images not found at {root}; use download=True.")
            items = [
                (path, int(class_dir.name))
                for class_dir in sorted(root.iterdir())
                if class_dir.is_dir() and class_dir.name.isdigit()
                for path in sorted(class_dir.glob("*.ppm"))
            ]
        else:
            root = self._test_root()
            gt_path = self._base_folder / "GT-final_test.csv"
            if not root.is_dir() or not gt_path.exists():
                raise FileNotFoundError(f"GTSRB test images/labels not found under {self._base_folder}.")
            with gt_path.open("r", encoding="utf-8") as handle:
                items = [
                    (root / row["Filename"], int(row["ClassId"]))
                    for row in csv.DictReader(handle, delimiter=";", skipinitialspace=True)
                ]
        images, targets = images_to_array(items, crop=None, label=f"GTSRB/{split}")
        return {"images": images, "labels": targets}
