"""Evaluation loop for the tuned lens model."""
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import sys
from typing import Dict, List, Literal, Optional, Tuple

import torch as th
import torch.nn as nn
from simple_parsing import field
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from tuned_lens.nn.lenses import Lens, LogitLens, TunedLens, TunedLensConfig
from tuned_lens.nn.unembed import Unembed
from tuned_lens.scripts.ingredients import (
    Data,
    Distributed,
    Model,
)
from tuned_lens.stats import LogitStats
from tuned_lens.utils import (
    maybe_all_reduce,
    pytree_map,
    pytree_stack,
    shift_labels,
    shift_preds,
)

LensType = Literal["logit", "tuned", "lora"]


logger = logging.getLogger(__name__)


def _nested_dict():
    return defaultdict(_nested_dict)


def _is_local_lens_dir(path: Path) -> bool:
    return (path / "config.json").exists() and (path / "params.pt").exists()


def _is_native_checkpoint_file(path: Path) -> bool:
    return path.is_file() and path.name.startswith("lens_step_") and path.suffix == ".pt"


def _is_native_run_dir(path: Path) -> bool:
    return path.is_dir() and any(path.glob("lens_step_*.pt"))


def _infer_tuned_artifact_d_model(state: Dict[str, th.Tensor]) -> Optional[int]:
    for key in sorted(state):
        value = state[key]
        if key.endswith(".weight") and value.ndim == 2:
            return int(value.shape[1])
    return None


def _load_local_tuned_lens(model: PreTrainedModel, lens_path: Path) -> TunedLens:
    """Load a local tuned-lens artifact, correcting stale converted configs."""
    with (lens_path / "config.json").open("r") as f:
        config_dict = json.load(f)

    state = th.load(lens_path / "params.pt", map_location="cpu")
    inferred_d_model = _infer_tuned_artifact_d_model(state)
    if inferred_d_model is not None and config_dict.get("d_model") != inferred_d_model:
        logger.warning(
            "Overriding d_model=%s from %s with d_model=%s inferred from params.pt",
            config_dict.get("d_model"),
            lens_path / "config.json",
            inferred_d_model,
        )
        config_dict["d_model"] = inferred_d_model

    config = TunedLensConfig.from_dict(config_dict)
    lens = TunedLens(Unembed(model), config)
    lens.layer_translators.load_state_dict(state)
    return lens


def _checkpoint_step_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    try:
        return (int(stem.rsplit("_", 1)[-1]), path.name)
    except (IndexError, ValueError):
        return (-1, path.name)


def _native_checkpoints_in_run(path: Path) -> List[Path]:
    return sorted(
        (p for p in path.glob("lens_step_*.pt") if p.is_file()),
        key=_checkpoint_step_key,
    )


def _native_checkpoints_in_tree(path: Path) -> List[Path]:
    checkpoints = []
    for child in sorted(p for p in path.iterdir() if p.is_dir()):
        checkpoints.extend(_native_checkpoints_in_run(child))
    return checkpoints


def _native_checkpoint_job_name(path: Path) -> str:
    return f"{path.parent.name}__{path.stem}"


def _native_checkpoint_output_dir(root: Path, checkpoint_path: Path) -> Path:
    return root / _native_checkpoint_job_name(checkpoint_path)


def _activation_sites_path(resource_path: Path) -> Optional[Path]:
    if resource_path.is_file():
        candidate = resource_path.parent / "activation_sites.csv"
    else:
        candidate = resource_path / "activation_sites.csv"
    return candidate if candidate.exists() else None


def _run_config_path(resource_path: Path) -> Optional[Path]:
    if resource_path.is_file():
        candidate = resource_path.parent / "run_config.csv"
    else:
        candidate = resource_path / "run_config.csv"
    return candidate if candidate.exists() else None


def _ensure_omnilens_importable() -> None:
    repo_src = Path(__file__).resolve().parents[3] / "omnilens" / "src"
    repo_src_str = str(repo_src)
    if repo_src.exists() and repo_src_str not in sys.path:
        sys.path.insert(0, repo_src_str)


class NativeLensAdapter(nn.Module):
    def __init__(self, native_lens: nn.Module, site_ids: List[str]):
        super().__init__()
        self.native_lens = native_lens
        self.site_ids = list(site_ids)

    def forward(self, h: th.Tensor, idx: int) -> th.Tensor:
        site_id = self.site_ids[idx]
        output = self.native_lens(h, layer=site_id)
        logits = getattr(output, "logits", None)
        if logits is None:
            raise ValueError(f"Native lens returned no logits for site {site_id!r}")
        return logits


def _load_native_checkpoint_metadata(resource_path: Path) -> Tuple[dict, List[str]]:
    checkpoint = th.load(resource_path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Native checkpoint {resource_path} is missing config metadata")

    activation_sites_path = _activation_sites_path(resource_path)
    if activation_sites_path is not None:
        with activation_sites_path.open(newline="") as f:
            site_ids = [row["site_id"] for row in csv.DictReader(f)]
    else:
        activation_preset = config.get("activation_site_preset", "residual")
        if activation_preset != "residual":
            raise ValueError(
                f"Native checkpoint {resource_path} needs activation_sites.csv for "
                f"activation_site_preset={activation_preset!r}"
            )
        num_layers = int(config.get("num_hidden_layers") or config.get("activation_site_count") or 0)
        if num_layers <= 0:
            raise ValueError(f"Cannot determine site IDs for native checkpoint {resource_path}")
        site_ids = [str(idx) for idx in range(num_layers)]

    return checkpoint, site_ids


def _load_native_lens(model: PreTrainedModel, resource_path: str) -> Tuple[str, nn.Module]:
    _ensure_omnilens_importable()
    from omnilens.lenses import create_lens
    from omnilens.training import HFUnembed, get_model_config

    checkpoint_path = Path(resource_path)
    checkpoint, site_ids = _load_native_checkpoint_metadata(checkpoint_path)
    config = checkpoint["config"]
    model_cfg = get_model_config(model)
    native_unembed = HFUnembed(model)

    lens_kwargs = {}
    lens_type = config["lens_type"]
    if lens_type == "lora":
        lens_kwargs["r"] = int(config["lora_rank"])
        lens_kwargs["alpha"] = float(config.get("lora_alpha", 1.0))
    elif lens_type == "tuned":
        lens_kwargs["bias"] = any(key.endswith(".bias") for key in checkpoint["lens_state_dict"])
    else:
        raise ValueError(f"Unsupported native lens type for eval: {lens_type!r}")

    native_lens = create_lens(
        lens_type,
        layer_ids=site_ids,
        hidden_size=model_cfg["hidden_size"],
        unembed=native_unembed,
        **lens_kwargs,
    )
    native_lens.load_checkpoint_state_dict(checkpoint["lens_state_dict"])
    return lens_type, NativeLensAdapter(native_lens, site_ids)


@dataclass
class ActivationSiteSpec:
    site_ids: List[str]
    hidden_state_sources: Dict[int, int]
    custom_hook_sources: Dict[int, str]
    custom_hook_paths: List[str]


class ActivationCapture:
    def __init__(self, model: PreTrainedModel, hook_paths: List[str]):
        self._captured = {}
        self._handles = []
        modules = dict(model.named_modules())
        for path in hook_paths:
            module = modules.get(path)
            if module is None:
                raise KeyError(f"Hook module not found in eval model: {path}")
            self._handles.append(module.register_forward_hook(self._make_hook(path)))

    def _make_hook(self, path: str):
        def hook(_module, _inputs, output):
            self._captured[path] = output

        return hook

    def clear(self) -> None:
        self._captured.clear()

    def get(self, path: str):
        return self._captured[path]

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()


def _normalize_activation(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"Expected a single activation tensor, got {type(value)}")
        return value[0]
    return value


def _resolve_hook_path(path: str, available_modules: set) -> str:
    if path in available_modules:
        return path

    stripped = path.replace("._fsdp_wrapped_module", "")
    if stripped in available_modules:
        return stripped

    raise KeyError(f"Could not resolve hook path {path!r} in eval model.")


def _load_activation_site_spec(
    model: PreTrainedModel,
    lens_name: Optional[str],
) -> Optional[ActivationSiteSpec]:
    if lens_name is None:
        return None

    lens_path = Path(lens_name)
    activation_sites_path = _activation_sites_path(lens_path)
    if activation_sites_path is None or not activation_sites_path.exists():
        return None

    available_modules = {name for name, _ in model.named_modules()}
    site_ids = []
    hidden_state_sources = {}
    custom_hook_sources = {}
    custom_hook_paths = []

    with activation_sites_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    for idx, row in enumerate(rows):
        site_id = row["site_id"]
        source_type = row["source_type"]
        source_ref = row["source_ref"]
        site_ids.append(site_id)

        if source_type == "hidden_state":
            hidden_state_sources[idx] = int(source_ref)
            continue

        if source_type != "custom_hook":
            raise ValueError(f"Unsupported activation source type {source_type!r}")

        resolved = _resolve_hook_path(source_ref, available_modules)
        custom_hook_sources[idx] = resolved
        custom_hook_paths.append(resolved)

    return ActivationSiteSpec(
        site_ids=site_ids,
        hidden_state_sources=hidden_state_sources,
        custom_hook_sources=custom_hook_sources,
        custom_hook_paths=sorted(set(custom_hook_paths)),
    )


@dataclass
class Eval:
    """Type hinting for CLI args."""

    data: Data

    model: Model

    dist: Distributed

    output: Path = field(alias=["-o"])
    """Folder to save the eval results to."""

    lens_name: Optional[str] = field(alias=["-l"], default=None)
    """Path to a tuned-lens artifact or native OmniLens checkpoint/run/tree."""

    logit: bool = True
    """Whether to evaluate the logit lens"""

    seed: int = 42
    """Random seed used for data shuffling."""

    tokens: Optional[int] = None
    """Number of tokens to evaluate on. If None, will use the entire dataset."""

    token_shift: int = field(default=1)
    """How to shift the labels wrt the input tokens (1 = next token, 0 = current token,
    -1 = previous token, etc.)"""

    per_gpu_batch_size: int = 1
    """Number of samples to try to fit on a GPU at once."""

    layer_transfer: bool = field(action="store_true")
    """Evaluate the transfer of the lens to different layers of the transformer."""

    record_logit_stats: bool = field(action="store_true")
    """Record the statistics of the marginal token distribution at each layer."""

    def load_lens(self, model: PreTrainedModel) -> Dict[str, nn.Module]:
        """Load the requested lens model."""
        lenses = {}
        if self.logit:
            lenses["logit"] = LogitLens.from_model(model)
        if self.lens_name is not None:
            lens_path = Path(self.lens_name)
            if _is_native_checkpoint_file(lens_path):
                lens_type, lens = _load_native_lens(model, self.lens_name)
                lenses[lens_type] = lens
            elif _is_local_lens_dir(lens_path):
                lenses["tuned"] = _load_local_tuned_lens(model, lens_path)
            else:
                lenses["tuned"] = TunedLens.from_model_and_pretrained(model, self.lens_name)
        return lenses

    def _resolve_eval_jobs(self) -> List[Tuple[str, Optional[str], Path]]:
        """Resolve one or more eval jobs from lens_name and output."""
        if self.lens_name is None:
            return [("logit_only", None, self.output)]

        lens_path = Path(self.lens_name)
        if not lens_path.exists():
            return [(self.lens_name, self.lens_name, self.output)]

        if _is_native_checkpoint_file(lens_path):
            return [(lens_path.stem, self.lens_name, self.output)]

        if _is_native_run_dir(lens_path):
            checkpoints = _native_checkpoints_in_run(lens_path)
            if len(checkpoints) == 1:
                checkpoint = checkpoints[0]
                return [(checkpoint.stem, str(checkpoint), self.output)]
            return [
                (
                    _native_checkpoint_job_name(checkpoint),
                    str(checkpoint),
                    _native_checkpoint_output_dir(self.output, checkpoint),
                )
                for checkpoint in checkpoints
            ]

        if _is_local_lens_dir(lens_path):
            return [(lens_path.name, self.lens_name, self.output)]

        native_tree = _native_checkpoints_in_tree(lens_path)
        if native_tree:
            return [
                (
                    _native_checkpoint_job_name(checkpoint),
                    str(checkpoint),
                    _native_checkpoint_output_dir(self.output, checkpoint),
                )
                for checkpoint in native_tree
            ]

        child_lenses = sorted(
            child for child in lens_path.iterdir() if child.is_dir() and _is_local_lens_dir(child)
        )
        if not child_lenses:
            raise ValueError(
                f"{self.lens_name} is neither a lens artifact directory nor a directory "
                "containing lens artifact subdirectories."
            )

        return [
            (child.name, str(child), self.output / child.name)
            for child in child_lenses
        ]

    def calculate_batch_limit(self, tokens_per_sample: int):
        """Calculate the total number of batches to evaluate on."""
        assert self.tokens is not None
        global_batch_size = self.dist.world_size * self.per_gpu_batch_size
        tokens_per_batch = global_batch_size * tokens_per_sample
        return self.tokens // tokens_per_batch

    def _initialize_logit_stats_recorders(
        self,
        lenses: Dict[str, nn.Module],
        site_names: List[str],
    ):
        if self.record_logit_stats:
            self.logit_stats_recorders = {
                lens_type: {site_name: LogitStats() for site_name in site_names}
                for lens_type in lenses.keys()
            }
            self.logit_stats_recorder_final = LogitStats()
        else:
            self.logit_stats_recorders = None
            self.logit_stats_recorder_final = None

    def _record_logit_stats(self, logp: th.Tensor, site_name: str, lens_type: str):
        if self.logit_stats_recorders is not None:
            self.logit_stats_recorders[lens_type][site_name].update(
                logp, assume_normalized=True
            )

    def _record_logit_stats_final(self, logp: th.Tensor):
        if self.logit_stats_recorder_final is not None:
            self.logit_stats_recorder_final.update(logp, assume_normalized=True)

    def _save_logit_stats(self) -> defaultdict:
        logit_stats = _nested_dict()
        if self.logit_stats_recorders is not None:
            for lens_type, recorders in self.logit_stats_recorders.items():
                for layer, recorder in recorders.items():
                    recorder.all_reduce_()
                    logit_stats[lens_type]["logit_stats"][layer] = (
                        recorder.marginal_probs.cpu().numpy().tolist()
                    )

        if self.logit_stats_recorder_final is not None:
            self.logit_stats_recorder_final.all_reduce_()
            logit_stats["baseline"]["logit_stats"]["final"] = (
                self.logit_stats_recorder_final.marginal_probs.cpu().numpy().tolist()
            )

        return logit_stats

    def _evaluate_lenses_on_hidden(
        self,
        lenses: Dict[str, nn.Module],
        hidden: th.Tensor,
        layer_idx: int,
        layer_name: str,
        final_probs: th.Tensor,
        final_lps: th.Tensor,
        labels: th.Tensor,
        batch_output: defaultdict,
        site_names: List[str],
    ):
        """Evaluate a lens at a given layer. Batch output is modified in place.

        Args:
            lenses: The dictionary of lenses to evaluate on this hidden state.
            hidden: (batch x seq x d_model) The hidden states of the transformer.
            layer: The layer this hidden state is from.
            final_probs: (batch x seq x vocab) The final probabilities of
                the transformer.
            final_lps: (batch x seq x vocab) The final log probabilities
                of the transformer.
            labels: (batch x seq) The labels for the transformer.
            batch_output: Where to store the logging results.
            total_layers: The total number of layers in the transformer.
            logp_stats: where to record the logging results.
        """
        for lens_type, lens in lenses.items():
            lens_lps = lens(hidden, idx=layer_idx).log_softmax(dim=-1)
            lens_probs = lens_lps.exp()

            self._record_logit_stats(lens_lps, layer_name, lens_type)

            batch_output[lens_type]["ce"][layer_name] = th.nn.functional.cross_entropy(
                shift_preds(lens_lps, self.token_shift).flatten(0, 1),
                labels.flatten(),
                reduction="none",
            )

            batch_output[lens_type]["entropy"][layer_name] = th.sum(
                -lens_probs * lens_lps, dim=-1
            )

            batch_output[lens_type]["kl"][layer_name] = th.sum(
                final_probs * (final_lps - lens_lps), dim=-1
            )

            if self.layer_transfer:
                for i, trans_name in enumerate(site_names):
                    transfer_lps = lens(hidden, idx=i).log_softmax(dim=-1)
                    batch_output[lens_type]["layer_transfer"]["ce"][trans_name][
                        layer_name
                    ] = th.nn.functional.cross_entropy(
                        shift_preds(transfer_lps, self.token_shift).flatten(0, 1),
                        labels.flatten(),
                    )
                    batch_output[lens_type]["layer_transfer"]["kl"][trans_name][
                        layer_name
                    ] = th.sum(lens_probs * (lens_lps - transfer_lps), dim=-1).mean()

    def _collect_site_activations(
        self,
        hidden_states,
        activation_capture: ActivationCapture,
        activation_site_spec: ActivationSiteSpec,
    ) -> List[Tuple[int, str, th.Tensor]]:
        ordered_sites = []
        for idx, site_id in enumerate(activation_site_spec.site_ids):
            if idx in activation_site_spec.hidden_state_sources:
                hidden_idx = activation_site_spec.hidden_state_sources[idx]
                tensor = hidden_states[hidden_idx]
            else:
                hook_path = activation_site_spec.custom_hook_sources[idx]
                tensor = activation_capture.get(hook_path)
            ordered_sites.append((idx, site_id, _normalize_activation(tensor)))
        return ordered_sites

    def _run_single_eval(
        self,
        model: PreTrainedModel,
        data,
        nats_to_bpb: float,
        lenses: Dict[str, nn.Module],
        root_dir: Path,
        job_name: str,
        activation_site_spec: Optional[ActivationSiteSpec],
    ):
        dl = self.dist.dataloader(data)
        dl.seed(self.seed)

        for lens in lenses.values():
            lens.eval()

        if self.tokens is not None:
            tokens_per_sample = len(data[0]["input_ids"])
            if self.tokens > len(data) * tokens_per_sample:
                raise ValueError(
                    f"Requested to evaluate on {self.tokens} tokens, "
                    f"but dataset only contains {len(data)*tokens_per_sample} tokens."
                )

            batch_limit = self.calculate_batch_limit(tokens_per_sample)
            assert batch_limit > 0, "Batch limit must be positive."
            dl = islice(dl, batch_limit)
            total = batch_limit
        else:
            total = len(data) // self.dist.world_size

        activation_capture = None
        if activation_site_spec is not None:
            site_names = activation_site_spec.site_ids
            activation_capture = ActivationCapture(model, activation_site_spec.custom_hook_paths)
        else:
            L = model.config.num_hidden_layers
            site_names = [f"layer_{i}" for i in range(L)]

        self._initialize_logit_stats_recorders(lenses, site_names)

        root_dir.mkdir(exist_ok=True, parents=True)
        batches = []

        try:
            self.dist.barrier()
            logger.info("Starting eval job '%s' on %s batches.", job_name, total)

            pbar = tqdm(dl, desc=f"Evaluating {job_name}", position=self.dist.rank, total=total)
            for batch in pbar:
                batch = self.dist.send_to_device(batch)
                if activation_capture is not None:
                    activation_capture.clear()
                output = model(**batch, output_hidden_states=True)

                final_lps = output.logits.log_softmax(dim=-1)
                final_probs = final_lps.exp()
                assert not th.isnan(output.logits).any(), "Logits are NaN"

                labels = shift_labels(batch["input_ids"], self.token_shift)
                batch_output = _nested_dict()

                if activation_site_spec is None:
                    hidden_states = output.hidden_states[:-1]
                    ordered_sites = [
                        (idx, site_names[idx], hidden)
                        for idx, hidden in enumerate(hidden_states)
                    ]
                else:
                    ordered_sites = self._collect_site_activations(
                        output.hidden_states,
                        activation_capture,
                        activation_site_spec,
                    )

                for layer_idx, layer_name, hidden in ordered_sites:
                    self._evaluate_lenses_on_hidden(
                        lenses=lenses,
                        hidden=hidden,
                        layer_idx=layer_idx,
                        layer_name=layer_name,
                        final_probs=final_probs,
                        final_lps=final_lps,
                        labels=labels,
                        batch_output=batch_output,
                        site_names=site_names,
                    )

                batch_output["baseline"]["ce"]["final"] = th.nn.functional.cross_entropy(
                    shift_preds(final_lps, self.token_shift).flatten(0, 1),
                    labels.flatten(),
                    reduction="none",
                )
                batch_output["baseline"]["entropy"]["final"] = th.sum(
                    -final_probs * final_lps, dim=-1
                )

                batches.append(pytree_map(th.mean, batch_output))  # type: ignore[arg-type]
                self._record_logit_stats_final(final_lps)

            pbar.close()
        finally:
            if activation_capture is not None:
                activation_capture.close()
        agg = pytree_map(lambda x: nats_to_bpb * x.mean(), pytree_stack(batches))
        agg = pytree_map(lambda x: maybe_all_reduce(x), agg)
        agg = pytree_map(lambda x: x.cpu().numpy().item(), agg)

        assert isinstance(agg, dict)

        batches = pytree_map(lambda x: nats_to_bpb * x, batches)
        batches = pytree_map(lambda x: maybe_all_reduce(x), batches)
        batches = pytree_map(lambda x: x.cpu().item(), batches)
        assert isinstance(batches, list)

        logit_stats = self._save_logit_stats()

        if self.dist.primary:
            with (root_dir / "batches.jsonl").open("w") as f:
                json.dump(batches, f)

            with (root_dir / "aggregate_metrics.json").open("w") as f:
                json.dump(agg, f)

            if self.record_logit_stats:
                with (root_dir / "logit_stats.json").open("w") as f:
                    json.dump(logit_stats, f)

    @th.autocast("cuda", enabled=th.cuda.is_available())
    @th.no_grad()
    def execute(self):
        """Evaluates a TunedLens model against a transformer on a dataset."""
        # Load model, tokenizer, data, and lens
        self.dist.init()
        model = tokenizer = data = lenses = nats_to_bpb = None

        # See comment in train_loop.py for why we do this
        load_device = self.dist.device if not self.dist.fsdp else None
        if self.dist.primary:
            # Let the primary processes populate the cache
            model, tokenizer = self.model.load(load_device)
            data, nats_to_bpb = self.data.load(tokenizer)

        self.dist.barrier()  # Wait for primary to finish filling the cache

        if not self.dist.primary:
            # Let the non-primary processes load from the cache
            model, tokenizer = self.model.load(load_device, must_use_cache=True)
            data, nats_to_bpb = self.data.load(tokenizer)
            lenses = None

        assert model and tokenizer and data and nats_to_bpb

        model = self.dist.shard_model(model)
        jobs = self._resolve_eval_jobs()

        self.dist.barrier()
        logger.info("All processes initialized. Running %s eval job(s).", len(jobs))

        for job_name, lens_name, root_dir in jobs:
            self.lens_name = lens_name
            lenses = self.load_lens(model)
            lenses = {name: lens.to(self.dist.device) for name, lens in lenses.items()}
            activation_site_spec = _load_activation_site_spec(model, lens_name)
            self._run_single_eval(
                model=model,
                data=data,
                nats_to_bpb=nats_to_bpb,
                lenses=lenses,
                root_dir=root_dir,
                job_name=job_name,
                activation_site_spec=activation_site_spec,
            )
