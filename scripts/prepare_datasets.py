"""Download the multi-dataset benchmark data and build the 32x32 caches.

Run this once, on a machine with internet access (on HoreKa: the login node),
before submitting any job. Compute jobs then only read the small cache files
(``<FLOWGUARD_DATA_ROOT>/<dataset>/cache32/*.npz``) and never touch the network.

    python scripts/prepare_datasets.py --datasets all
    python scripts/prepare_datasets.py --datasets CelebA,LFW --remove-raw

Download sizes (approximate): CIFAR10/100 0.3 GB, GTSRB 0.3 GB, BelgiumTS
0.3 GB, LFW 0.2 GB, CelebA 11.7 GB (parquet), ISIC 2018 2.8 GB, ISIC 2019 9.1 GB.
Decoding CelebA takes ~10-20 min single-threaded; everything else is faster.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

import defenses.config as cfg  # noqa: E402
from defenses import datasets as legacy_datasets  # noqa: E402
from defenses.datasets.cached32 import Cached32Dataset, remove_raw_files  # noqa: E402

# Role of each dataset in the benchmark (victim task or attacker-side pool).
ALL_DATASETS: dict[str, str] = {
    "CIFAR10": "victim",
    "CIFAR100": "attacker pool for CIFAR10",
    "GTSRB": "victim",
    "BelgiumTS": "attacker pool for GTSRB",
    "CelebA": "victim",
    "LFW": "attacker pool for CelebA",
    "LFW10": "attacker proxy-classifier data for CelebA",
    "SkinCancer": "victim",
    "BCN20000": "attacker pool for SkinCancer",
}

EXPECTED_CLASSES = {
    "CIFAR10": 10, "CIFAR100": 100, "GTSRB": 43, "BelgiumTS": 62, "CelebA": 2,
    "SkinCancer": 7, "BCN20000": 8,
}


def prepare(name: str, *, download: bool) -> list[str]:
    dataset_cls = legacy_datasets.__dict__[name]
    lines = []
    for train in (True, False):
        started = time.time()
        dataset = dataset_cls(train=train, download=download)
        labels = np.asarray(dataset.targets)
        counts = np.bincount(labels, minlength=len(dataset.classes)) if len(labels) else []
        split = "train" if train else "test"
        lines.append(
            f"{name:11s} {split:5s} n={len(dataset):7d} classes={len(dataset.classes):4d} "
            f"min/max per class={int(np.min(counts)) if len(counts) else 0}/"
            f"{int(np.max(counts)) if len(counts) else 0} ({time.time() - started:.0f}s)"
        )
        expected = EXPECTED_CLASSES.get(name)
        if expected is not None and len(dataset.classes) != expected:
            raise RuntimeError(f"{name}: expected {expected} classes, got {len(dataset.classes)}.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="all",
                        help=f"Comma-separated subset of {','.join(ALL_DATASETS)} or 'all'.")
    parser.add_argument("--no-download", action="store_true",
                        help="Only build caches from raw files that are already in place.")
    parser.add_argument("--remove-raw", action="store_true",
                        help="Delete downloaded archives/images of cached datasets afterwards "
                             "(keeps only cache32/). Not applied to CIFAR, which is tiny.")
    args = parser.parse_args()

    names = list(ALL_DATASETS) if args.datasets == "all" else [n.strip() for n in args.datasets.split(",") if n.strip()]
    unknown = [name for name in names if name not in ALL_DATASETS]
    if unknown:
        raise SystemExit(f"Unknown datasets: {unknown}. Choose from {list(ALL_DATASETS)}.")

    print(f"[prepare] data root: {cfg.DATASET_ROOT}")
    report: list[str] = []
    failures: list[str] = []
    prepared: list[str] = []
    for name in names:
        print(f"[prepare] === {name} ({ALL_DATASETS[name]}) ===", flush=True)
        try:
            report.extend(prepare(name, download=not args.no_download))
            prepared.append(name)
        except Exception as error:  # noqa: BLE001 - keep preparing the others
            failures.append(f"{name}: {error!r}")
            print(f"[prepare] FAILED {name}: {error!r}", flush=True)

    if args.remove_raw:
        # After the loop: LFW and LFW10 read the same raw folder.
        failed = {line.split(":", 1)[0] for line in failures}
        folders_done: set[str] = set()
        for name in prepared:
            dataset_cls = legacy_datasets.__dict__[name]
            if not (isinstance(dataset_cls, type) and issubclass(dataset_cls, Cached32Dataset)):
                continue
            sharing = [
                other for other in ALL_DATASETS
                if getattr(legacy_datasets.__dict__[other], "folder", None) == dataset_cls.folder
            ]
            cache_dir = Path(cfg.DATASET_ROOT) / dataset_cls.folder / "cache32"
            all_cached = all(
                (cache_dir / f"{other}_{split}.npz").exists()
                for other in sharing
                for split in ("train", "test")
            )
            if dataset_cls.folder in folders_done or not all_cached or any(o in failed for o in sharing):
                print(f"[prepare] keeping raw files of {dataset_cls.folder} (not every user is cached)")
                continue
            remove_raw_files(dataset_cls)
            folders_done.add(dataset_cls.folder)

    print("\n[prepare] summary")
    for line in report:
        print("  " + line)
    if failures:
        print("\n[prepare] failures")
        for line in failures:
            print("  " + line)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
