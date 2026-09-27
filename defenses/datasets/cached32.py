"""32x32 image datasets for the multi-dataset benchmark (CIFAR10/GTSRB/CelebA/Skin Cancer).

Every dataset in this module is decoded once, resized to 32x32 RGB and stored
as a single uint8 array in ``<DATASET_ROOT>/<folder>/cache32/<Class>_<split>.npz``.
After that, loading is one ``np.load`` instead of decoding tens of thousands of
small JPEG files per epoch, which matters on a parallel file system.

The resulting objects mimic torchvision's CIFAR10: ``data`` is an ``(N, 32, 32,
3)`` uint8 array, ``targets`` a list of ints and ``classes`` a list. The legacy
transfer-set code stores ``queryset.data[i]`` as raw images, so this layout is
what keeps every attack in the repository working unchanged.

The 32x32 resolution follows the FDINet evaluation protocol (Yao et al.), which
uses the same four dataset/architecture pairs and resizes every task to 32x32.

Dataset roles:

- Victim (defended) tasks: :class:`CelebA` (gender), :class:`SkinCancer`
  (ISIC 2018 Task 3 / HAM10000). GTSRB lives in ``gtsrb.py``.
- Attacker-side public pools (disjoint from the victim data): :class:`LFW`
  (faces, for CelebA), :class:`BelgiumTS` / :class:`TSRD` (traffic signs, for
  GTSRB), :class:`BCN20000` (dermoscopy, for Skin Cancer). :class:`LFW10` is the
  labelled LFW subset used to train the attacker's proxy classifier.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import defenses.config as cfg

IMAGE_SIZE = 32
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_array32(image: Image.Image, crop: int | str | None = None) -> np.ndarray:
    """Convert one PIL image to a 32x32x3 uint8 array.

    ``crop`` is ``None`` (resize the full frame), ``"square"`` (centre square of
    side ``min(w, h)``) or an int (centre square of that side, clamped to the
    image). PIL's bicubic resize low-pass filters when downsampling, so large
    images do not alias.
    """
    image = image.convert("RGB")
    if crop is not None:
        width, height = image.size
        side = min(width, height) if crop == "square" else min(int(crop), width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def stable_fraction(key: str) -> float:
    """Deterministic value in [0, 1) derived from ``key`` (for reproducible splits)."""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0x100000000)


def download_file(url: str, destination: Path) -> Path:
    """Download ``url`` to ``destination`` unless it already exists."""
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    print(f"[download] {url} -> {destination}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "flowguard-dataset-prep"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        copied = 0
        next_report = 256 * 1024 * 1024
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            copied += len(chunk)
            if copied >= next_report:
                print(f"[download]   {copied / 1e9:.2f} GB", flush=True)
                next_report += 256 * 1024 * 1024
    partial.replace(destination)
    return destination


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract a zip/tar archive into ``destination``."""
    archive = Path(archive)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive.name} -> {destination}", flush=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
        return
    with tarfile.open(archive) as handle:
        try:
            handle.extractall(destination, filter="data")
        except TypeError:  # Python without tarfile extraction filters
            handle.extractall(destination)


def find_file(root: Path, name: str) -> Path | None:
    """Return the first file called ``name`` below ``root``."""
    if not root.exists():
        return None
    direct = root / name
    if direct.is_file():
        return direct
    for candidate in root.rglob(name):
        if candidate.is_file():
            return candidate
    return None


def find_dir_containing(root: Path, filename: str) -> Path | None:
    """Return the directory below ``root`` that contains ``filename``."""
    found = find_file(root, filename)
    return found.parent if found is not None else None


def _progress(done: int, total: int, label: str) -> None:
    if done % 5000 == 0 or done == total:
        print(f"[cache32] {label}: {done}/{total}", flush=True)


def images_to_array(
    items: list[tuple[Path, int]],
    *,
    crop: int | str | None,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode ``(path, label)`` pairs into stacked arrays."""
    images = np.empty((len(items), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    labels = np.empty((len(items),), dtype=np.int64)
    for index, (path, target) in enumerate(items):
        with Image.open(path) as image:
            images[index] = to_array32(image, crop)
        labels[index] = int(target)
        _progress(index + 1, len(items), label)
    return images, labels


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Cached32Dataset(Dataset):
    """CIFAR-like dataset backed by a cached 32x32 uint8 array.

    Subclasses set ``folder`` and ``classes`` and implement ``_build(split)``,
    which returns a dict with at least ``images`` (N, 32, 32, 3) uint8 and
    ``labels`` (N,) int64. They may override ``download``.
    """

    folder: str = ""
    classes: list[Any] = []
    splits: tuple[str, ...] = ("train", "test")

    def __init__(
        self,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
        split: str | None = None,
    ) -> None:
        self.root = Path(cfg.DATASET_ROOT) / self.folder
        self.split = split or ("train" if train else "test")
        if self.split not in self.splits:
            raise ValueError(f"{type(self).__name__} has no split '{self.split}' ({self.splits}).")
        self.train = self.split == "train"
        self.transform = transform
        self.target_transform = target_transform

        cache_path = self.cache_path(self.split)
        if not cache_path.exists():
            if download:
                self.download()
            payload = self._build(self.split)
            self._write_cache(cache_path, payload)
        with np.load(cache_path, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        if "class_names" in arrays:
            # Restored from the cache so num_classes survives deleting raw files.
            self.classes = [str(name) for name in arrays["class_names"]]
        self.data = arrays["images"]
        self.targets = [int(value) for value in self._labels_from_payload(arrays)]
        if len(self.targets) != len(self.data):
            raise RuntimeError(f"{cache_path}: {len(self.data)} images but {len(self.targets)} labels.")

    # -- cache ----------------------------------------------------------------

    def cache_path(self, split: str) -> Path:
        return self.root / "cache32" / f"{type(self).__name__}_{split}.npz"

    @staticmethod
    def _write_cache(path: Path, payload: dict[str, np.ndarray]) -> None:
        if payload["images"].ndim != 4 or payload["images"].shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"Unexpected cached image shape {payload['images'].shape}.")
        if len(payload["images"]) == 0:
            raise RuntimeError(f"Refusing to write an empty cache: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temporary name so concurrent builders never interleave writes.
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
        temporary.replace(path)
        print(f"[cache32] wrote {path} ({len(payload['images'])} images)", flush=True)

    def _labels_from_payload(self, payload: dict[str, np.ndarray]) -> np.ndarray:
        return payload["labels"]

    # -- subclass hooks -------------------------------------------------------

    def download(self) -> None:
        """Fetch raw files. Default: nothing to download (manual placement)."""

    def _build(self, split: str) -> dict[str, np.ndarray]:
        raise NotImplementedError

    # -- Dataset API ----------------------------------------------------------

    def __len__(self) -> int:
        return int(len(self.data))

    def __getitem__(self, index: int):
        image = Image.fromarray(self.data[index])
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def get_image(self, index: int) -> np.ndarray:
        return self.data[index]


# ---------------------------------------------------------------------------
# CelebA (victim task: gender)
# ---------------------------------------------------------------------------


CELEBA_HF_REPO = "flwrlabs/celeba"
CELEBA_HF_CONFIG = "img_align+identity+attr"


class CelebA(Cached32Dataset):
    """CelebA aligned faces, binary gender task (attribute ``Male``), 32x32.

    Follows FDINet, which uses the gender attribute. Other attributes can be
    selected with ``FLOWGUARD_CELEBA_ATTR`` (the cache stores all 40 attribute
    columns, so switching attributes does not rebuild it).

    Splits are the official ones: train 162,770 / valid 19,867 / test 19,962.
    Images are centre-cropped to 148x148 (face region of the 178x218 aligned
    image) before resizing, the usual CelebA preprocessing for small inputs.

    Accepted raw layouts under ``<DATASET_ROOT>/celeba``:

    1. HuggingFace parquet shards (what ``download=True`` fetches, from the
       ungated ``flwrlabs/celeba`` mirror): ``hf/img_align+identity+attr/*.parquet``.
    2. The official release: ``img_align_celeba/`` + ``list_attr_celeba.txt`` +
       ``list_eval_partition.txt`` (Kaggle's ``.csv`` variants also work).
    """

    folder = "celeba"
    splits = ("train", "valid", "test")
    crop = 148

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.target_attribute = os.environ.get("FLOWGUARD_CELEBA_ATTR", "Male")
        if self.target_attribute == "Male":
            self.classes = ["female", "male"]
        else:
            self.classes = [f"not_{self.target_attribute}", self.target_attribute]
        super().__init__(*args, **kwargs)

    def _labels_from_payload(self, payload: dict[str, np.ndarray]) -> np.ndarray:
        names = [str(name) for name in payload["attr_names"]]
        if self.target_attribute not in names:
            raise KeyError(f"CelebA attribute '{self.target_attribute}' not in {names}.")
        column = names.index(self.target_attribute)
        return (payload["attributes"][:, column] > 0).astype(np.int64)

    # -- download -------------------------------------------------------------

    def download(self) -> None:
        if self._official_layout() is not None or self._parquet_files("train"):
            return
        api = f"https://huggingface.co/api/datasets/{CELEBA_HF_REPO}"
        with urllib.request.urlopen(api, timeout=60) as response:
            listing = json.load(response)
        files = [
            entry["rfilename"]
            for entry in listing.get("siblings", [])
            if entry["rfilename"].startswith(CELEBA_HF_CONFIG + "/")
            and entry["rfilename"].endswith(".parquet")
        ]
        if not files:
            raise RuntimeError(f"No parquet shards listed for {CELEBA_HF_REPO}.")
        for name in files:
            url = (
                f"https://huggingface.co/datasets/{CELEBA_HF_REPO}/resolve/main/"
                + urllib.parse.quote(name)
            )
            download_file(url, self.root / "hf" / name)

    # -- build ----------------------------------------------------------------

    def _parquet_files(self, split: str) -> list[Path]:
        shard_dir = self.root / "hf" / CELEBA_HF_CONFIG
        return sorted(shard_dir.glob(f"{split}-*.parquet")) if shard_dir.exists() else []

    def _official_layout(self) -> tuple[Path, Path, Path] | None:
        image_dir = find_dir_containing(self.root, "000001.jpg")
        attr = find_file(self.root, "list_attr_celeba.txt") or find_file(self.root, "list_attr_celeba.csv")
        part = find_file(self.root, "list_eval_partition.txt") or find_file(
            self.root, "list_eval_partition.csv"
        )
        if image_dir is None or attr is None or part is None:
            return None
        return image_dir, attr, part

    def _build(self, split: str) -> dict[str, np.ndarray]:
        shards = self._parquet_files(split)
        if shards:
            return self._build_from_parquet(shards, split)
        official = self._official_layout()
        if official is not None:
            return self._build_from_official(*official, split=split)
        raise FileNotFoundError(
            f"CelebA not found under {self.root}. Run with download=True "
            "(scripts/prepare_datasets.py --datasets CelebA) or place the official "
            "img_align_celeba/ + list_attr_celeba.txt + list_eval_partition.txt there."
        )

    def _build_from_parquet(self, shards: list[Path], split: str) -> dict[str, np.ndarray]:
        import pyarrow.parquet as pq

        attr_names: list[str] | None = None
        images: list[np.ndarray] = []
        attributes: list[np.ndarray] = []
        for shard in shards:
            parquet = pq.ParquetFile(shard)
            if attr_names is None:
                attr_names = [
                    name for name in parquet.schema_arrow.names if name not in {"image", "celeb_id"}
                ]
            for batch in parquet.iter_batches(batch_size=512, columns=["image", *attr_names]):
                columns = batch.to_pydict()
                for row_index, cell in enumerate(columns["image"]):
                    raw = cell["bytes"] if isinstance(cell, dict) else cell
                    with Image.open(io.BytesIO(raw)) as image:
                        images.append(to_array32(image, self.crop))
                    attributes.append(
                        np.asarray(
                            [1 if columns[name][row_index] else -1 for name in attr_names],
                            dtype=np.int8,
                        )
                    )
                    _progress(len(images), -1, f"CelebA/{split}")
        return {
            "images": np.stack(images),
            "labels": np.zeros(len(images), dtype=np.int64),
            "attributes": np.stack(attributes),
            "attr_names": np.asarray(attr_names),
        }

    def _build_from_official(
        self, image_dir: Path, attr_path: Path, part_path: Path, *, split: str
    ) -> dict[str, np.ndarray]:
        split_code = {"train": 0, "valid": 1, "test": 2}[split]

        def rows(path: Path) -> Iterable[list[str]]:
            with path.open("r", encoding="utf-8") as handle:
                if path.suffix == ".csv":
                    yield from csv.reader(handle)
                else:
                    for line in handle:
                        yield line.split()

        partition: dict[str, int] = {}
        for row in rows(part_path):
            if len(row) >= 2 and row[1].strip().lstrip("-").isdigit():
                partition[row[0]] = int(row[1])

        attr_rows = list(rows(attr_path))
        if attr_path.suffix == ".csv":
            attr_names = attr_rows[0][1:]
            body = attr_rows[1:]
        else:
            attr_names = attr_rows[1]
            body = attr_rows[2:]
        filenames = [row[0] for row in body if partition.get(row[0]) == split_code]
        attr_lookup = {row[0]: row[1:] for row in body}
        images = np.empty((len(filenames), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        attributes = np.empty((len(filenames), len(attr_names)), dtype=np.int8)
        for index, name in enumerate(filenames):
            with Image.open(image_dir / name) as image:
                images[index] = to_array32(image, self.crop)
            attributes[index] = np.asarray([int(value) for value in attr_lookup[name]], dtype=np.int8)
            _progress(index + 1, len(filenames), f"CelebA/{split}")
        return {
            "images": images,
            "labels": np.zeros(len(filenames), dtype=np.int64),
            "attributes": attributes,
            "attr_names": np.asarray(attr_names),
        }


# ---------------------------------------------------------------------------
# ISIC 2018 Task 3 / HAM10000 (victim task) and BCN20000 (attacker pool)
# ---------------------------------------------------------------------------


ISIC_S3 = "https://isic-challenge-data.s3.amazonaws.com"


def _read_isic_ground_truth(path: Path, classes: list[str]) -> dict[str, int]:
    """Map image id -> class index from an ISIC one-hot ground-truth CSV."""
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores = [float(row[name]) for name in classes]
            labels[row["image"]] = int(np.argmax(scores))
    return labels


def _index_images(root: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in root.rglob("*")
        if path.suffix.lower() in _IMAGE_EXTENSIONS
    }


class SkinCancer(Cached32Dataset):
    """ISIC 2018 Task 3 (= HAM10000) dermoscopy, 7 diagnostic classes, 32x32.

    This is the "Skin Cancer" task of FDINet (reference: Codella et al., ISIC
    2018 challenge). Train = the 10,015 official training images; test = the
    1,512 official test images, which come from different lesions, so there is
    no lesion leakage between the two. ``split='valid'`` gives the 193 official
    validation images. Images (600x450) are centre-cropped to a square first.
    """

    folder = "isic2018"
    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    splits = ("train", "valid", "test")
    _FILES = {
        "train": ("ISIC2018_Task3_Training_Input.zip", "ISIC2018_Task3_Training_GroundTruth.zip"),
        "valid": ("ISIC2018_Task3_Validation_Input.zip", "ISIC2018_Task3_Validation_GroundTruth.zip"),
        "test": ("ISIC2018_Task3_Test_Input.zip", "ISIC2018_Task3_Test_GroundTruth.zip"),
    }
    _GT_CSV = {
        "train": "ISIC2018_Task3_Training_GroundTruth.csv",
        "valid": "ISIC2018_Task3_Validation_GroundTruth.csv",
        "test": "ISIC2018_Task3_Test_GroundTruth.csv",
    }

    def _split_dir(self, split: str) -> Path:
        return self.root / split

    def download(self) -> None:
        for split, archives in self._FILES.items():
            target = self._split_dir(split)
            if find_file(target, self._GT_CSV[split]) is not None:
                continue
            for archive in archives:
                local = download_file(f"{ISIC_S3}/2018/{archive}", self.root / "archives" / archive)
                extract_archive(local, target)

    def _build(self, split: str) -> dict[str, np.ndarray]:
        directory = self._split_dir(split)
        gt_path = find_file(directory, self._GT_CSV[split])
        if gt_path is None:
            raise FileNotFoundError(
                f"ISIC 2018 {split} ground truth not found under {directory}. Run with "
                "download=True (scripts/prepare_datasets.py --datasets SkinCancer)."
            )
        labels = _read_isic_ground_truth(gt_path, self.classes)
        index = _index_images(directory)
        missing = [name for name in labels if name not in index]
        if missing:
            raise FileNotFoundError(f"{len(missing)} ISIC 2018 {split} images missing, e.g. {missing[:3]}")
        items = [(index[name], labels[name]) for name in sorted(labels)]
        images, targets = images_to_array(items, crop="square", label=f"SkinCancer/{split}")
        return {"images": images, "labels": targets}


class BCN20000(Cached32Dataset):
    """BCN20000 dermoscopy images (the ``BCN_`` lesions of ISIC 2019), 8 classes.

    Attacker-side public pool for the Skin Cancer task, as in FDINet. Filtering
    on the ``BCN_`` lesion prefix excludes every HAM10000 image (prefix
    ``HAM_``), so the pool is disjoint from the victim's training data. The
    12,413 images (3,576 lesions) are split 90/10 *by lesion* so that no lesion
    appears in both splits.
    """

    folder = "isic2019"
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    _INPUT = "ISIC_2019_Training_Input.zip"
    _META = "ISIC_2019_Training_Metadata.csv"
    _GT = "ISIC_2019_Training_GroundTruth.csv"

    def download(self) -> None:
        for name in (self._META, self._GT):
            download_file(f"{ISIC_S3}/2019/{name}", self.root / name)
        if find_dir_containing(self.root / "images", "ISIC_0053454.jpg") is None:
            local = download_file(f"{ISIC_S3}/2019/{self._INPUT}", self.root / "archives" / self._INPUT)
            extract_archive(local, self.root / "images")

    def _build(self, split: str) -> dict[str, np.ndarray]:
        meta_path = self.root / self._META
        gt_path = self.root / self._GT
        if not meta_path.exists() or not gt_path.exists():
            raise FileNotFoundError(
                f"ISIC 2019 metadata not found under {self.root}. Run with download=True "
                "(scripts/prepare_datasets.py --datasets BCN20000)."
            )
        labels = _read_isic_ground_truth(gt_path, self.classes)
        index = _index_images(self.root / "images")
        items: list[tuple[Path, int]] = []
        with meta_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                lesion = row.get("lesion_id") or ""
                if not lesion.startswith("BCN_"):
                    continue
                in_test = stable_fraction(lesion) < 0.10
                if (split == "test") != in_test:
                    continue
                image_id = row["image"]
                if image_id not in index:
                    raise FileNotFoundError(f"BCN20000 image {image_id} missing under {self.root}/images")
                items.append((index[image_id], labels[image_id]))
        items.sort(key=lambda item: item[0].name)
        images, targets = images_to_array(items, crop="square", label=f"BCN20000/{split}")
        return {"images": images, "labels": targets}


# ---------------------------------------------------------------------------
# LFW (attacker pool for CelebA)
# ---------------------------------------------------------------------------


LFW_URLS = (
    "https://ndownloader.figshare.com/files/5976015",  # lfw-funneled.tgz (figshare mirror)
    "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz",
)


class LFW(Cached32Dataset):
    """Labeled Faces in the Wild (funneled), all 13,233 images, 32x32.

    Attacker-side public face pool for CelebA, as in FDINet. The central 150x150
    region of the 250x250 frame is kept so the framing roughly matches the
    CelebA crop. Labels are identity indices; the split is a deterministic 90/10
    split per image. Use :class:`LFW10` when a well-posed classification task is
    needed.
    """

    folder = "lfw"
    crop = 150
    min_faces_per_person = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.root = Path(cfg.DATASET_ROOT) / self.folder
        self.classes = self._people()
        super().__init__(*args, **kwargs)

    def download(self) -> None:
        if self._image_root() is not None:
            return
        archive = self.root / "lfw-funneled.tgz"
        last_error: Exception | None = None
        for url in LFW_URLS:
            try:
                download_file(url, archive)
                break
            except Exception as error:  # noqa: BLE001 - try the next mirror
                last_error = error
        else:
            raise RuntimeError(f"Could not download LFW: {last_error}")
        extract_archive(archive, self.root)

    def _image_root(self) -> Path | None:
        candidate = self.root / "lfw_funneled"
        return candidate if candidate.is_dir() else None

    def _people(self) -> list[str]:
        image_root = self._image_root()
        if image_root is None:
            # Classes are only needed for num_classes; resolved again after download.
            return []
        people = []
        for person in sorted(path for path in image_root.iterdir() if path.is_dir()):
            count = sum(1 for _ in person.glob("*.jpg"))
            if count >= self.min_faces_per_person:
                people.append(person.name)
        return people

    def _build(self, split: str) -> dict[str, np.ndarray]:
        image_root = self._image_root()
        if image_root is None:
            raise FileNotFoundError(
                f"LFW not found under {self.root}. Run with download=True "
                "(scripts/prepare_datasets.py --datasets LFW)."
            )
        self.classes = self._people()
        items: list[tuple[Path, int]] = []
        for class_index, person in enumerate(self.classes):
            for path in sorted((image_root / person).glob("*.jpg")):
                in_test = stable_fraction(path.name) < 0.10
                if (split == "test") == in_test:
                    items.append((path, class_index))
        images, targets = images_to_array(items, crop=self.crop, label=f"{type(self).__name__}/{split}")
        return {"images": images, "labels": targets, "class_names": np.asarray(self.classes)}


class LFW10(LFW):
    """LFW people with at least 10 images (158 identities, ~4.3k images).

    Only used to train the attacker's proxy classifier, whose gradients produce
    the PGD pairs for the attacker's surrogate FlowPure CNF.
    """

    min_faces_per_person = 10


# ---------------------------------------------------------------------------
# Traffic-sign pools for GTSRB
# ---------------------------------------------------------------------------


BELGIUM_URLS = {
    "train": "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip",
    "test": "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip",
}


class BelgiumTS(Cached32Dataset):
    """Belgian Traffic Sign Classification benchmark (62 classes), 32x32.

    Attacker-side public traffic-sign pool for GTSRB. FDINet used the Chinese
    TSRD set, whose download server is frequently unreachable; BelgiumTS is a
    European traffic-sign set that is disjoint from GTSRB and downloads
    reliably. :class:`TSRD` is available if the files are placed manually.
    """

    folder = "belgiumts"
    classes = list(range(62))

    def _split_root(self, split: str) -> Path:
        return self.root / ("Training" if split == "train" else "Testing")

    def download(self) -> None:
        for split, url in BELGIUM_URLS.items():
            if self._split_root(split).is_dir():
                continue
            archive = download_file(url, self.root / Path(url).name)
            extract_archive(archive, self.root)

    def _build(self, split: str) -> dict[str, np.ndarray]:
        split_root = self._split_root(split)
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"BelgiumTS not found at {split_root}. Run with download=True "
                "(scripts/prepare_datasets.py --datasets BelgiumTS)."
            )
        items = [
            (path, int(class_dir.name))
            for class_dir in sorted(split_root.iterdir())
            if class_dir.is_dir() and class_dir.name.isdigit()
            for path in sorted(class_dir.glob("*.ppm"))
        ]
        images, targets = images_to_array(items, crop=None, label=f"BelgiumTS/{split}")
        return {"images": images, "labels": targets}


class TSRD(Cached32Dataset):
    """Chinese Traffic Sign Recognition Database (58 classes), manual placement.

    Download ``tsrd-train.zip`` and ``TSRD-Test.zip`` from
    http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html and extract them
    to ``<DATASET_ROOT>/tsrd/train`` and ``<DATASET_ROOT>/tsrd/test``. File names
    start with the class id (``000_0001.png``).
    """

    folder = "tsrd"
    classes = list(range(58))

    def _build(self, split: str) -> dict[str, np.ndarray]:
        split_root = self.root / split
        if not split_root.is_dir():
            raise FileNotFoundError(
                f"TSRD not found at {split_root}. Download it manually from "
                "http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html (see class docstring)."
            )
        items = [
            (path, int(path.name.split("_", 1)[0]))
            for path in sorted(split_root.rglob("*"))
            if path.suffix.lower() in _IMAGE_EXTENSIONS and path.name.split("_", 1)[0].isdigit()
        ]
        images, targets = images_to_array(items, crop=None, label=f"TSRD/{split}")
        return {"images": images, "labels": targets}


def remove_raw_files(dataset_cls: type[Cached32Dataset]) -> None:
    """Delete everything except the 32x32 caches (frees space after preparation)."""
    root = Path(cfg.DATASET_ROOT) / dataset_cls.folder
    for child in root.iterdir():
        if child.name == "cache32":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
