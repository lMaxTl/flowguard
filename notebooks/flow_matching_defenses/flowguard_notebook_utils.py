from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

try:
    import torch
except ImportError:  # Summary plots still work without torch.
    torch = None


# Wong (2011) colorblind-safe palette for Nature-style publication figures.
NATURE = {
    "white": "#FFFFFF",
    "black": "#000000",
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#999999",
}


ATTACKS = [
    "a1_clean_transfer",
    "a2_projected_maze",
    "a3_flowblind",
    "adaptive_adaptive",
    "maze_baseline",
]


ATTACK_EXPLANATIONS = {
    "a1_clean_transfer": (
        "**A1 Clean-Transfer** queries a clean external image pool. The attack is "
        "dangerous for input-anomaly defenses because every individual input can "
        "look natural even though the query stream is used for model stealing."
    ),
    "a2_projected_maze": (
        "**A2 ProjectedMAZE** runs MAZE but denoises each generated query before "
        "submission. This suppresses high-frequency artifacts that per-query "
        "FlowPure-style scores often detect."
    ),
    "a3_flowblind": (
        "**A3 FlowBlind** adds a surrogate-CNF velocity penalty to the MAZE "
        "generator. It tries to make generated images low-scoring under the "
        "same signal used by FlowPure."
    ),
    "adaptive_adaptive": (
        "**Adaptive-Adaptive** combines A2 denoising with a stronger A3 "
        "surrogate-CNF penalty. It is the strongest inline attack in this suite "
        "because it targets both pixel artifacts and the detector score."
    ),
    "maze_baseline": (
        "**MAZE baseline** is unmodified generator-based extraction. It usually "
        "produces off-manifold or repetitive queries and should be easiest for "
        "FlowPure-like geometry detectors to catch."
    ),
}


DEFENSE_EXPLANATIONS = {
    "flowguard_integral": (
        "**FlowGuard Integral (C1)** replaces the single `t=0` velocity score "
        "with an integral of squared velocity along a short reverse ODE path. "
        "An attack that only minimizes the start-point score can still be "
        "energetic elsewhere on the trajectory."
    ),
    "flowguard_userlevel": (
        "**FlowGuard User-Level (C3)** computes the normal per-query FlowPure "
        "score but evaluates the user's sliding-window score distribution with "
        "a KS statistic against benign reference traffic. It is meant for "
        "stealing behavior that is weak per query but suspicious over time."
    ),
    "flowguard_labelhist": (
        "**FlowGuard Label-Histogram (C4)** is an output-side detector. It "
        "tracks victim top-1 labels and compares the user histogram with a "
        "benign reference. It catches attacks that query strategically across "
        "classes while keeping inputs visually clean."
    ),
}


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by walking up to `pyproject.toml`."""
    root = (start or Path.cwd()).resolve()
    while not (root / "pyproject.toml").exists() and root != root.parent:
        root = root.parent
    return root


def setup_notebook_environment(start: Path | None = None) -> tuple[Path, Path, Path]:
    """Register repository paths so notebook kernels can import ``defenses`` and ``flowguard``."""
    notebook_dir = (start or Path.cwd()).resolve()
    if (notebook_dir / "flowguard_notebook_utils.py").exists():
        utils_dir = notebook_dir
    else:
        utils_dir = notebook_dir / "notebooks" / "flow_matching_defenses"
    project_root = find_project_root(notebook_dir)
    src_root = project_root / "src"
    for path in (project_root, src_root, utils_dir):
        token = str(path)
        if token not in sys.path:
            sys.path.insert(0, token)
    return project_root, src_root, utils_dir


def apply_nature_style() -> None:
    """Apply matplotlib defaults suitable for Nature-style publication figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": NATURE["white"],
            "axes.facecolor": NATURE["white"],
            "savefig.facecolor": NATURE["white"],
            "axes.edgecolor": NATURE["black"],
            "axes.labelcolor": NATURE["black"],
            "text.color": NATURE["black"],
            "xtick.color": NATURE["black"],
            "ytick.color": NATURE["black"],
            "grid.color": NATURE["gray"],
            "grid.alpha": 0.3,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_suite_rows(run_root: Path) -> list[dict[str, Any]]:
    """Load consolidated FlowGuard++ suite rows, falling back to pair metrics."""
    summary_path = run_root / "flowguard_suite_summary.json"
    if summary_path.exists():
        return [dict(row) for row in load_json(summary_path).get("results", [])]

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(run_root.glob("*/attack_suite_metrics.json")):
        pair_name = metrics_path.parent.name
        if "__" not in pair_name:
            continue
        defense, _attack = pair_name.split("__", 1)
        payload = load_json(metrics_path)
        for row in payload.get("results", []):
            merged = dict(row)
            merged["defense"] = defense
            rows.append(merged)
    return rows


def defense_rows(run_root: Path, defense: str) -> list[dict[str, Any]]:
    """Return result rows for a single defense in attack order."""
    rows = [row for row in load_suite_rows(run_root) if row.get("defense") == defense]
    order = {attack: index for index, attack in enumerate(ATTACKS)}
    return sorted(rows, key=lambda row: order.get(str(row.get("attack")), 999))


def display_defense_intro(defense: str) -> None:
    """Render a compact defense and attack explanation block."""
    attack_lines = "\n".join(
        f"- {ATTACK_EXPLANATIONS[attack]}" for attack in ATTACKS if attack in ATTACK_EXPLANATIONS
    )
    display(
        Markdown(
            f"{DEFENSE_EXPLANATIONS[defense]}\n\n"
            "### Attack catalogue\n"
            f"{attack_lines}"
        )
    )


def display_metric_table(rows: list[dict[str, Any]]) -> None:
    """Render a Markdown table with the important suite metrics."""
    if not rows:
        display(Markdown("_No rows found for this defense._"))
        return
    lines = [
        "| attack | AUROC | FPR | detection rate | substitute accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        auroc = row.get("auroc")
        fpr = row.get("fpr")
        detection_rate = row.get("detection_rate")
        lines.append(
            "| {attack} | {auroc:.4f} | {fpr:.4f} | {det:.4f} | {acc} |".format(
                attack=row.get("attack", "unknown"),
                auroc=float("nan") if auroc is None else float(auroc),
                fpr=float("nan") if fpr is None else float(fpr),
                det=float("nan") if detection_rate is None else float(detection_rate),
                acc=(
                    "n/a"
                    if row.get("substitute_accuracy") is None
                    else f"{float(row['substitute_accuracy']):.2f}"
                ),
            )
        )
    display(Markdown("\n".join(lines)))


def plot_defense_metrics(rows: list[dict[str, Any]], defense: str) -> None:
    """Plot AUROC, FPR, and detection rate for one defense."""
    if not rows:
        print(f"No rows found for {defense}.")
        return
    attacks = [str(row["attack"]) for row in rows]
    x = np.arange(len(attacks), dtype=np.float64)
    width = 0.26
    auroc = np.asarray([
        np.nan if row.get("auroc") is None else float(row["auroc"])
        for row in rows
    ])
    fpr = np.asarray([
        np.nan if row.get("fpr") is None else float(row["fpr"])
        for row in rows
    ])
    det = np.asarray([
        np.nan if row.get("detection_rate") is None else float(row["detection_rate"])
        for row in rows
    ])

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - width, auroc, width, label="AUROC", color=NATURE["blue"])
    ax.bar(x, det, width, label="attack detection rate", color=NATURE["green"])
    ax.bar(x + width, fpr, width, label="FPR", color=NATURE["orange"])
    ax.axhline(0.5, color=NATURE["gray"], linestyle=":", linewidth=1.0, label="random AUROC")
    ax.set_title(f"{defense}: detection metrics by attack")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, labels=attacks, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_utility_vs_detection(rows: list[dict[str, Any]], defense: str) -> None:
    """Show the tradeoff between attack utility and detector success."""
    points = [
        row for row in rows
        if row.get("substitute_accuracy") is not None and row.get("detection_rate") is not None
    ]
    if not points:
        print("No substitute accuracy values were reported for this defense.")
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for row in points:
        x = float(row["detection_rate"])
        y = float(row["substitute_accuracy"])
        ax.scatter(x, y, s=70, color=NATURE["green"])
        ax.text(x + 0.01, y, str(row["attack"]), va="center", fontsize=9)
    ax.set_title(f"{defense}: attack utility vs detection")
    ax.set_xlabel("attack detection rate")
    ax.set_ylabel("substitute accuracy")
    ax.set_xlim(-0.02, 1.05)
    fig.tight_layout()
    plt.show()


def pair_dir(run_root: Path, defense: str, attack: str) -> Path:
    """Return the suite directory for a defense/attack pair."""
    return run_root / f"{defense}__{attack}"


def experiment_result_path(run_root: Path, defense: str, attack: str, *, benign: bool = False) -> Path | None:
    """Find the attack or benign experiment_result.json for a pair."""
    prefix = "benign_reference" if benign else f"attack-{attack}"
    root = pair_dir(run_root, defense, attack)
    matches = sorted(root.glob(f"{prefix}*/experiment_result.json"))
    return matches[0] if matches else None


def extract_scores_from_experiment(path: Path) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Extract mirrored `flowpure_scores`/blocked flags from experiment JSON."""
    payload = load_json(path)
    scores: list[float] = []
    flags: list[float] = []
    threshold: float | None = None
    records = payload.get("query_summary", {}).get("history", {}).get("records", [])
    for record in records:
        metadata = record.get("metadata", {}) or {}
        raw_scores = metadata.get("flowpure_scores", [])
        raw_flags = metadata.get("flowpure_blocked", [])
        scores.extend(float(value) for value in raw_scores)
        flags.extend(1.0 if value else 0.0 for value in raw_flags[: len(raw_scores)])
        if threshold is None and "flowpure_threshold" in metadata:
            threshold = float(metadata["flowpure_threshold"])
    return np.asarray(scores, dtype=np.float64), np.asarray(flags, dtype=np.float64), threshold


def plot_score_distributions(run_root: Path, defense: str, attacks: Iterable[str] = ATTACKS) -> None:
    """Plot benign vs attack score distributions for every available attack."""
    attack_list = list(attacks)
    fig, axes = plt.subplots(len(attack_list), 1, figsize=(8, 2.2 * len(attack_list)), sharex=False)
    axes_array = np.asarray(axes).reshape(-1)
    for ax, attack in zip(axes_array, attack_list):
        attack_path = experiment_result_path(run_root, defense, attack, benign=False)
        benign_path = experiment_result_path(run_root, defense, attack, benign=True)
        if attack_path is None or benign_path is None:
            ax.set_title(f"{attack}: missing experiment_result.json")
            continue
        attack_scores, _attack_flags, threshold = extract_scores_from_experiment(attack_path)
        benign_scores, _benign_flags, _ = extract_scores_from_experiment(benign_path)
        if attack_scores.size == 0 or benign_scores.size == 0:
            ax.set_title(f"{attack}: no scores recorded")
            continue
        ax.hist(benign_scores, bins=30, alpha=0.65, label="benign", color=NATURE["blue"])
        ax.hist(attack_scores, bins=30, alpha=0.65, label="attack", color=NATURE["red"])
        if threshold is not None:
            ax.axvline(threshold, color=NATURE["black"], linestyle="--", label="threshold")
        ax.set_title(f"{attack}: score distribution")
        ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


def build_single_defense_command(
    project_root: Path,
    run_root: Path,
    defense: str,
    *,
    attack_query_budget: int = 248,
    benign_query_budget: int = 248,
    device: str = "cuda",
    force: bool = False,
) -> list[str]:
    """Build the evaluate_flowguard_suite.py command for one defense."""
    python = project_root / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = Path(sys.executable)
    command = [
        str(python),
        str(project_root / "scripts" / "evaluate_flowguard_suite.py"),
        "--output-root", str(run_root),
        "--defenses", defense,
        "--attacks", ",".join(ATTACKS),
        "--dataset", "CIFAR10",
        "--query-dataset", "CIFAR10",
        "--clean-transfer-dataset", "CIFAR100",
        "--target-checkpoint-dir",
        str(project_root / "runs" / "notebook" / "training-victim-cifar10-vgg16_bn-nodefense" / "target_model"),
        "--flow-checkpoint",
        str(project_root / "runs" / "flow_matching" / "cifar10_flowpure_pgd" / "checkpoint_latest.pt"),
        "--a3-surrogate-checkpoint",
        str(project_root / "runs" / "flow_matching" / "cifar10_flowpure_pgd" / "checkpoint_latest.pt"),
        "--device", device,
        "--target-fpr", "0.05",
        "--attack-query-budget", str(attack_query_budget),
        "--benign-query-budget", str(benign_query_budget),
        "--attack-batch-size", "1",
        "--extraction-attack-batch-size", "16",
        "--training-batch-size", "16",
        "--epochs", "1",
        "--seed-samples", "16",
    ]
    if force:
        command.append("--force")
    return command


def run_command(command: list[str], project_root: Path) -> None:
    """Print and execute a command from the repository root."""
    print(" ".join(shlex.quote(str(part)) for part in command))
    subprocess.run(command, cwd=project_root, check=True)


def load_torch_payload(path: Path) -> Any:
    """Load a torch artifact with compatibility for older torch versions."""
    if torch is None:
        raise ImportError("torch is required for loading visualization artifacts.")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_attack_artifact(run_root: Path, defense: str, attack: str, filename: str) -> Path | None:
    """Find an attack artifact below a defense/attack pair directory."""
    matches = sorted(pair_dir(run_root, defense, attack).glob(f"attack-{attack}*/attack/{filename}"))
    return matches[0] if matches else None


def query_tensor_from_record(record: Any) -> Any:
    """Extract the submitted query tensor from a visualization/transfer record."""
    if isinstance(record, dict):
        return record.get("query_x", record.get("x"))
    if isinstance(record, (tuple, list)) and record:
        return record[0]
    raise TypeError(f"Unsupported record type: {type(record)!r}")


def to_cifar_image_array(value: Any) -> np.ndarray:
    """Convert a CIFAR tensor/array to HWC [0, 1] for matplotlib."""
    if torch is not None:
        tensor = torch.as_tensor(value).detach().cpu().float()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dimensions, got {tuple(tensor.shape)}.")

        is_hwc = tensor.shape[-1] == 3
        if is_hwc:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] != 3:
            raise ValueError(f"Expected CHW or HWC RGB image, got {tuple(tensor.shape)}.")

        tensor_min = float(tensor.min())
        tensor_max = float(tensor.max())
        if tensor_min >= 0.0 and tensor_max > 1.1:
            # Clean-transfer artifacts can be stored as uint8-like 0..255 images.
            tensor = tensor / 255.0
        elif tensor_min < -0.1 or tensor_max > 1.1:
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            tensor = tensor * std + mean
        return tensor.clamp(0, 1).permute(1, 2, 0).numpy()

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if float(array.min(initial=0.0)) >= 0.0 and float(array.max(initial=0.0)) > 1.1:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def show_attack_query_grid(run_root: Path, defense: str, attack: str, *, count: int = 12) -> None:
    """Show a grid of query images submitted by one attack."""
    artifact = find_attack_artifact(run_root, defense, attack, "visualization_records.pt")
    if artifact is None:
        artifact = find_attack_artifact(run_root, defense, attack, "transferset.pickle")
    if artifact is None:
        print(f"No query artifact found for {defense} x {attack}.")
        return
    records = load_torch_payload(artifact)
    if not records:
        print(f"Artifact is empty: {artifact}")
        return

    selected = list(records[: min(count, len(records))])
    columns = min(6, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(1.8 * columns, 1.9 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for ax, record in zip(axes_array, selected):
        ax.imshow(to_cifar_image_array(query_tensor_from_record(record)))
        ax.set_axis_off()
    for ax in axes_array[len(selected):]:
        ax.set_axis_off()
    fig.suptitle(f"{defense} x {attack}: submitted queries")
    fig.tight_layout()
    plt.show()
    print(f"Loaded: {artifact}")


def _label_from_record(record: Any) -> int | None:
    if isinstance(record, dict):
        label = record.get("label", record.get("y"))
    elif isinstance(record, (tuple, list)) and len(record) > 1:
        label = record[1]
    else:
        return None

    if torch is not None:
        tensor = torch.as_tensor(label).detach().cpu()
        if tensor.numel() == 0:
            return None
        if tensor.numel() == 1:
            return int(tensor.item())
        return int(torch.argmax(tensor.flatten()).item())
    array = np.asarray(label)
    if array.size == 0:
        return None
    if array.size == 1:
        return int(array.item())
    return int(np.argmax(array.reshape(-1)))


def label_histogram_from_artifact(path: Path, *, num_classes: int = 10) -> np.ndarray:
    """Read top-1 labels from a transfer/visualization artifact."""
    records = load_torch_payload(path)
    labels = [_label_from_record(record) for record in records]
    labels = [label for label in labels if label is not None]
    if not labels:
        return np.zeros(num_classes, dtype=np.float64)
    histogram = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    return histogram / max(float(histogram.sum()), 1.0)


def rbf_mmd(a: np.ndarray, b: np.ndarray, *, sigma: float = 1.0) -> float:
    """Unbiased RBF-MMD^2 between two histograms."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return 0.0
    diff_aa = a[:, None] - a[None, :]
    diff_bb = b[:, None] - b[None, :]
    diff_ab = a[:, None] - b[None, :]
    k_aa = np.exp(-(diff_aa ** 2) / (2.0 * sigma ** 2))
    k_bb = np.exp(-(diff_bb ** 2) / (2.0 * sigma ** 2))
    k_ab = np.exp(-(diff_ab ** 2) / (2.0 * sigma ** 2))
    return float(
        (k_aa.sum() - np.trace(k_aa)) / (a.size * (a.size - 1))
        + (k_bb.sum() - np.trace(k_bb)) / (b.size * (b.size - 1))
        - 2.0 * k_ab.mean()
    )


def plot_label_histogram_analysis(run_root: Path, *, source_defense: str = "flowpure") -> None:
    """Evaluate the C4 label-histogram signal post-hoc on suite artifacts."""
    rows: list[dict[str, Any]] = []
    benign_reference: np.ndarray | None = None
    for attack in ATTACKS:
        benign_artifact = sorted(
            pair_dir(run_root, source_defense, attack).glob("benign_reference*/attack/transferset.pickle")
        )
        if benign_artifact:
            benign_reference = label_histogram_from_artifact(benign_artifact[0])
            break
    if benign_reference is None:
        print("No benign transfer-set artifact found for label-histogram analysis.")
        return

    for attack in ATTACKS:
        artifact = find_attack_artifact(run_root, source_defense, attack, "transferset.pickle")
        if artifact is None:
            continue
        histogram = label_histogram_from_artifact(artifact)
        rows.append({"attack": attack, "histogram": histogram, "mmd": rbf_mmd(histogram, benign_reference)})

    if not rows:
        print("No attack transfer-set artifacts found for label-histogram analysis.")
        return

    labels = [row["attack"] for row in rows]
    values = [float(row["mmd"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(np.arange(len(values)), values, color=NATURE["green"])
    ax.set_title("C4 post-hoc MMD vs benign label histogram")
    ax.set_ylabel("RBF-MMD")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=30, ha="right")
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 1.8 * len(rows)), sharex=True)
    axes_array = np.asarray(axes).reshape(-1)
    class_ids = np.arange(benign_reference.size)
    for ax, row in zip(axes_array, rows):
        ax.bar(class_ids - 0.18, benign_reference, width=0.36, label="benign", color=NATURE["blue"])
        ax.bar(class_ids + 0.18, row["histogram"], width=0.36, label=row["attack"], color=NATURE["orange"])
        ax.set_title(f"{row['attack']} label histogram, MMD={row['mmd']:.5f}")
        ax.set_ylabel("frequency")
        ax.legend(loc="upper right")
    axes_array[-1].set_xlabel("victim top-1 class")
    fig.tight_layout()
    plt.show()
