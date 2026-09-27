from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

import defenses.models.zoo as legacy_zoo
from defenses import datasets as legacy_datasets
from defenses.utils import model as legacy_model_utils
from flowguard.attacks.base import AttackRunContext
from flowguard.attacks.registry import build_attack_runner
from flowguard.defenses.prediction.adaptive_misinformation import (
    AdaptiveMisinformationDefense,
)
from flowguard.defenses.prediction.base import (
    IdentityPredictionDefense,
    PredictionContext,
)
from flowguard.defenses.prediction.mad import MadDefense
from flowguard.defenses.prediction.modelguard import ModelGuardDefense
from flowguard.defenses.prediction.quantization import QuantizationDefense
from flowguard.defenses.prediction.random_noise import RandomNoiseDefense
from flowguard.defenses.prediction.reverse_sigmoid import ReverseSigmoidDefense
from flowguard.defenses.query.budgeting import BudgetingQueryDefense
from flowguard.defenses.query.fdinet import FDINetQueryDefense
from flowguard.defenses.query.flow_matching import FlowMatchingQueryDefense
from flowguard.defenses.query.flowguard import (
    FlowGuardCompositeDefense,
    FlowGuardIntegralDefense,
    FlowGuardLabelHistogramDefense,
    FlowGuardUserLevelDefense,
)
from flowguard.defenses.query.flowpure import FlowPureQueryDefense
from flowguard.defenses.query.multi import MultiAuditQueryDefense
from flowguard.defenses.query.noop import NoOpQueryDefense
from flowguard.defenses.query.prada import PradaQueryDefense
from flowguard.distributed.coordinator import DistributedCoordinator
from flowguard.distributed.worker import build_in_process_worker
from flowguard.evaluation.metrics import accuracy, joint_accuracy
from flowguard.evaluation.reports import build_evaluation_summary
from flowguard.experiments.spec import ExperimentSpec, QueryDefenseSpec
from flowguard.orchestration.results import (
    AttackSummary,
    ExperimentResult,
    QuerySummary,
)
from flowguard.querying.engine import QueryEngine
from flowguard.serving.model_loader import load_legacy_model
from flowguard.serving.target_service import TargetService
from flowguard.training.checkpoints import load_checkpoint
from flowguard.training.eval_datasets import build_eval_testset
from flowguard.training.substitute import train_substitute_model
from flowguard.training.target import train_target_model
from flowguard.training.victim_eval import wrap_victim_for_substitute_eval
from flowguard.workflows.misinformation import run_misinformation_workflow
from flowguard.workflows.outlier_exposure import run_outlier_exposure_workflow
from flowguard.workflows.proxy_models import train_proxy_model
from flowguard.workflows.shadow_models import train_shadow_models


def _log(spec: ExperimentSpec, message: str) -> None:
    if spec.verbose:
        print(f"[FlowGuard++] {message}")


def _result_to_payload(result: ExperimentResult) -> dict[str, object]:
    return {
        "experiment_name": result.experiment_name,
        "query_summary": asdict(result.query_summary),
        "attack_summary": asdict(result.attack_summary),
        "evaluation_summary": asdict(result.evaluation_summary),
        "artifacts": result.artifacts,
        "metadata": result.metadata,
    }


def _build_prediction_defense(
    spec: ExperimentSpec,
    loaded_model,
    prep_artifacts: dict[str, str] = None,
    target_checkpoint_dir: Path | None = None,
):
    name = spec.prediction_defense.name.lower()
    params = dict(spec.prediction_defense.parameters)
    if name in {"", "none"}:
        defense = IdentityPredictionDefense()
    elif name in {"reverse_sigmoid", "revsig"}:
        defense = ReverseSigmoidDefense(**params)
    elif name in {"random_noise", "rand_noise"}:
        defense = RandomNoiseDefense(**params)
    elif name == "mad":
        defense = MadDefense(**params)
    elif name in {"mld", "modelguard", "modelguard_w"}:
        defense = ModelGuardDefense(**params)
    elif name in {"am", "adaptive_misinformation"}:
        if prep_artifacts and "misinformation_model" in prep_artifacts:
            params["model_def_path"] = str(Path(prep_artifacts["misinformation_model"]) / "checkpoint.pth.tar")
        defense = AdaptiveMisinformationDefense(**params)
    elif name in {"quantization", "modelguard_s"}:
        params["trainingset_name"] = spec.dataset.name
        if target_checkpoint_dir is not None:
            params["out_path"] = str(target_checkpoint_dir)
        defense = QuantizationDefense(**params)
    else:
        raise ValueError(f"Unsupported prediction defense: {spec.prediction_defense.name}")
    defense.prepare(PredictionContext(loaded_model=loaded_model, defense_name=name, parameters=params))
    return [defense]


def _build_query_defenses(spec: ExperimentSpec, loaded_model=None):
    name = spec.query_defense.name.lower()
    params = dict(spec.query_defense.parameters)
    if name in {"multi_audit", "multi"}:
        # Audit several detectors against one query stream. Children are built
        # through this same function so every detector keeps its normal
        # construction path (dataset/device defaults, loaded_model wiring).
        children: list[tuple[str, Any]] = []
        child_identities: dict[str, int] = {}
        for entry in params.get("defenses", []):
            if entry.get("sybil_identities") is not None:
                child_identities[str(entry.get("key", entry["name"]))] = int(entry["sybil_identities"])
            child_spec = replace(
                spec,
                query_defense=QueryDefenseSpec(
                    name=str(entry["name"]),
                    parameters=dict(entry.get("parameters", {})),
                ),
            )
            for built in _build_query_defenses(child_spec, loaded_model=loaded_model):
                children.append((str(entry.get("key", entry["name"])), built))
        return [
            MultiAuditQueryDefense(
                children,
                strict=bool(params.get("strict", False)),
                child_identities=child_identities,
            )
        ]
    if name in {"", "noop", "none"}:
        return [NoOpQueryDefense()]
    if name in {"budgeting", "budget"}:
        return [
            BudgetingQueryDefense(
                budget=spec.attack.query_budget,
                **params,
            )
        ]
    if name == "prada":
        return [PradaQueryDefense(**params)]
    if name in {"fdinet", "fdi"}:
        return [FDINetQueryDefense(loaded_model=loaded_model, **params)]
    if name in {"flow_matching", "flowmatching", "fm"}:
        params.setdefault("dataset_name", spec.dataset.name)
        params.setdefault("device", spec.target_model.device)
        return [FlowMatchingQueryDefense(**params)]
    if name == "flowpure":
        params.setdefault("dataset_name", spec.dataset.name)
        params.setdefault("device", spec.target_model.device)
        return [FlowPureQueryDefense(**params)]
    if name in {"flowguard_integral", "flowguard++_integral", "c1"}:
        params.setdefault("dataset_name", spec.dataset.name)
        params.setdefault("device", spec.target_model.device)
        return [FlowGuardIntegralDefense(**params)]
    if name in {"flowguard_userlevel", "flowguard++_userlevel", "c3"}:
        params.setdefault("dataset_name", spec.dataset.name)
        params.setdefault("device", spec.target_model.device)
        return [FlowGuardUserLevelDefense(**params)]
    if name in {"flowguard_labelhist", "flowguard++_labelhist", "c4"}:
        return [FlowGuardLabelHistogramDefense(**params)]
    if name in {"flowguard++", "flowguardpp", "flowguard_composite", "hybrid"}:
        params.setdefault("dataset_name", spec.dataset.name)
        params.setdefault("device", spec.target_model.device)
        return [FlowGuardCompositeDefense(**params)]
    raise ValueError(f"Unsupported query defense: {spec.query_defense.name}")


def _ensure_target_checkpoint(spec: ExperimentSpec, run_root: Path) -> Path:
    target_dir = run_root / "target_model"
    if spec.target_model.checkpoint_dir:
        checkpoint_dir = Path(spec.target_model.checkpoint_dir)
        if checkpoint_dir.exists():
            if spec.preparation.outlier_exposure or spec.preparation.misinformation_training:
                if (checkpoint_dir / "model_poison.pt").exists():
                    return checkpoint_dir
            else:
                return checkpoint_dir
        else:
            target_dir = checkpoint_dir
    if spec.preparation.outlier_exposure or spec.preparation.misinformation_training:
        oe_dataset = spec.metadata.get("oe_dataset", spec.dataset.name)
        oe_lambda = float(spec.metadata.get("oe_lambda", 0.0))
        return run_outlier_exposure_workflow(
            spec,
            output_dir=target_dir,
            oe_dataset_name=oe_dataset,
            oe_lambda=oe_lambda,
        )
    return train_target_model(spec, output_dir=target_dir)


def _run_preparation_workflows(spec: ExperimentSpec, run_root: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if spec.preparation.proxy_training:
        proxy_dir = train_proxy_model(spec, output_dir=run_root / "proxy_model")
        artifacts["proxy_model"] = str(proxy_dir)
    if spec.preparation.shadow_training:
        shadow_dirs = train_shadow_models(
            spec,
            dataset_name=spec.attack.query_dataset.name,
            output_dir=run_root / "shadow_models",
            num_shadows=int(spec.metadata.get("num_shadows", 1)),
            num_classes=spec.metadata.get("shadow_num_classes"),
        )
        artifacts["shadow_models"] = str(run_root / "shadow_models")
        artifacts["shadow_model_count"] = str(len(shadow_dirs))
    if spec.preparation.misinformation_training and not spec.preparation.outlier_exposure:
        oe_dataset = spec.metadata.get("oe_dataset", spec.dataset.name)
        oe_lambda = float(spec.metadata.get("oe_lambda", 0.0))
        misinfo_dir = run_misinformation_workflow(
            spec,
            output_dir=run_root / "misinformation_model",
            oe_dataset_name=oe_dataset,
            oe_lambda=oe_lambda,
        )
        artifacts["misinformation_model"] = str(misinfo_dir)
    return artifacts


def _evaluate_substitute(spec: ExperimentSpec, substitute_dir: Path, target_model: torch.nn.Module):
    testset = build_eval_testset(spec)
    modelfamily = legacy_datasets.dataset_to_modelfamily[spec.dataset.name]
    eval_victim = wrap_victim_for_substitute_eval(target_model, spec)
    checkpoint_path = next(substitute_dir.glob("checkpoint*.pth.tar"), None)
    if checkpoint_path is None:
        return build_evaluation_summary({})
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=torch.device(spec.substitute_model.device),
    )
    model = legacy_zoo.get_net(
        spec.substitute_model.architecture,
        modelfamily,
        spec.substitute_model.pretrained,
        num_classes=len(testset.classes),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(torch.device(spec.substitute_model.device))
    loader = torch.utils.data.DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=spec.training.num_workers,
    )
    _, acc, fid = legacy_model_utils.test_step(
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        torch.device(spec.substitute_model.device),
        gt_model=eval_victim,
        silent=True,
    )
    all_predictions = []
    all_targets = []
    all_victim_predictions = []
    victim_device = next(target_model.parameters()).device
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(torch.device(spec.substitute_model.device))
            preds = model(inputs)
            victim_inputs = inputs.to(victim_device)
            victim = eval_victim(victim_inputs) if eval_victim is not None else target_model(victim_inputs)
            all_predictions.append(preds.detach().cpu())
            all_targets.append(targets.cpu())
            all_victim_predictions.append(victim.detach().cpu())
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    victim_predictions = torch.cat(all_victim_predictions, dim=0)
    return build_evaluation_summary(
        {
            "accuracy": accuracy(predictions, targets),
            "fidelity": fid,
            "joint_accuracy": joint_accuracy(predictions, targets, victim_predictions),
        }
    )


def run_experiment(spec: ExperimentSpec, output_dir: str | Path | None = None) -> ExperimentResult:
    spec.validate()
    run_root = Path(output_dir or Path("runs") / spec.name)
    run_root.mkdir(parents=True, exist_ok=True)
    _log(
        spec,
        (
            f"Starting experiment '{spec.name}' "
            f"(attack={spec.attack.kind.value}/{spec.attack.mode.name}, "
            f"defense={spec.prediction_defense.name}, distributed={spec.distributed.enabled})"
        ),
    )
    target_checkpoint_dir = _ensure_target_checkpoint(spec, run_root)
    _log(spec, f"Target checkpoint ready at '{target_checkpoint_dir}'")
    prep_artifacts = _run_preparation_workflows(spec, run_root)
    if prep_artifacts:
        _log(spec, f"Preparation workflows produced artifacts: {sorted(prep_artifacts.keys())}")
    loaded_model = load_legacy_model(target_checkpoint_dir, device=spec.target_model.device)
    prediction_defenses = _build_prediction_defense(
        spec,
        loaded_model,
        prep_artifacts,
        target_checkpoint_dir=target_checkpoint_dir,
    )
    query_defenses = _build_query_defenses(spec, loaded_model=loaded_model)
    target_service = TargetService(loaded_model, prediction_defenses=prediction_defenses)
    # In the non-distributed matrix path we emulate a Sybil attacker by spreading
    # queries across `num_clients` identities inside a single engine, so stateful
    # per-user defenses lose evidence per identity.
    sybil_identities = getattr(spec.distributed, "num_clients", 1) if not spec.distributed.enabled else 1
    query_engine = QueryEngine(
        target_service,
        query_defenses=query_defenses,
        sybil_num_identities=sybil_identities,
        sybil_granularity=str(spec.metadata.get("sybil_granularity", "query")),
    )

    distributed = None
    if spec.distributed.enabled:
        num_clients = getattr(spec.distributed, "num_clients", spec.distributed.num_workers)
        workers = [
            build_in_process_worker(
                QueryEngine(
                    target_service,
                    query_defenses=_build_query_defenses(spec, loaded_model=loaded_model),
                ),
                worker_id=f"worker-{idx}",
            )
            for idx in range(max(1, num_clients))
        ]
        distributed = DistributedCoordinator(
            workers,
            max_inflight_batches=spec.distributed.max_inflight_batches,
            num_threads=spec.distributed.num_workers
        )

    attack_runner = build_attack_runner(spec.attack.kind, query_engine=query_engine, distributed_coordinator=distributed)
    _log(spec, f"Running attack runner '{attack_runner.name}'")
    attack_result = attack_runner.run(
        AttackRunContext(
            experiment=spec,
            output_dir=str(run_root / "attack"),
            metadata={"shadow_models": prep_artifacts.get("shadow_models")},
        )
    )

    artifacts = dict(prep_artifacts)
    artifacts["target_model"] = str(target_checkpoint_dir)
    substitute_dir = Path(attack_result.output_dir)
    transferset_path_raw = attack_result.metadata.get("transferset_path")
    if transferset_path_raw:
        transferset_path = Path(transferset_path_raw)
        with transferset_path.open("rb") as handle:
            transfer_samples = load_checkpoint(handle)
        substitute_dir = Path(run_root / "substitute_model")
        if transfer_samples:
            _log(spec, f"Training substitute from transfer set '{transferset_path}'")
            train_substitute_model(
                spec,
                transfer_samples,
                output_dir=substitute_dir,
                victim_model=loaded_model.model,
            )
        else:
            # An attack can legitimately end up with zero transfer samples --
            # e.g. a rejection-oracle attack (accepted_only=True) against a
            # defense that detects every single query never accepts anything
            # to learn from. That is a real evaluation outcome (the defense
            # won outright), not a crash: samples_to_transferset would raise
            # IndexError on budget_samples[0][0] for an empty list, so skip
            # training entirely and let _evaluate_substitute's existing
            # "no checkpoint found" path report empty/N-A metrics instead.
            _log(
                spec,
                f"Skipping substitute training: transfer set '{transferset_path}' "
                "has zero samples (every query was rejected/blocked).",
            )
        artifacts["transfer_set"] = str(transferset_path)
    _log(spec, f"Evaluating substitute model from '{substitute_dir}'")
    evaluation_summary = _evaluate_substitute(spec, substitute_dir, loaded_model.model)
    if distributed is None:
        total_queries = query_engine.history.total_queries
        total_batches = query_engine.history.total_batches
        average_batch_size = query_engine.history.average_batch_size
        combined_history = query_engine.history
    else:
        total_batches = len(distributed.task_results)
        total_queries = sum(int(result.metadata.get("batch_size", 0)) for result in distributed.task_results)
        average_batch_size = (total_queries / total_batches) if total_batches else 0.0
        from flowguard.querying.history import QueryHistory
        combined_history = QueryHistory()
        for worker in workers:
            combined_history.merge(worker.query_engine.history)

    query_summary = QuerySummary(
        total_queries=total_queries,
        distinct_inputs=total_queries,
        average_batch_size=average_batch_size,
        total_batches=total_batches,
        history=combined_history,
    )
    result = ExperimentResult(
        experiment_name=spec.name,
        query_summary=query_summary,
        attack_summary=AttackSummary(
            attack_name=attack_result.attack_name,
            mode_name=attack_result.mode_name,
            output_dir=attack_result.output_dir,
            metadata=dict(attack_result.metadata),
        ),
        evaluation_summary=evaluation_summary,
        artifacts={**artifacts, "substitute_model": str(substitute_dir)},
        metadata={"distributed": spec.distributed.enabled},
    )
    with (run_root / "experiment_result.json").open("w", encoding="utf-8") as handle:
        json.dump(_result_to_payload(result), handle, indent=2)
    _log(
        spec,
        (
            f"Finished experiment '{spec.name}' "
            f"with total_queries={query_summary.total_queries} "
            f"and metrics={evaluation_summary.metrics}"
        ),
    )
    return result


def run_experiment_suite(
    specs: list[ExperimentSpec],
    *,
    output_root: str | Path | None = None,
) -> list[ExperimentResult]:
    root = Path(output_root or "runs")
    root.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []
    for spec in specs:
        _log(spec, f"Running suite member '{spec.name}'")
        result = run_experiment(spec, output_dir=root / spec.name)
        results.append(result)
    with (root / "suite_results.json").open("w", encoding="utf-8") as handle:
        json.dump([_result_to_payload(result) for result in results], handle, indent=2)
    return results
