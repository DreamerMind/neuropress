"""Memory-bounded block calibration with aligned float and student streams."""

from __future__ import annotations

import copy
import hashlib
import json
import weakref
from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from .calibration.batches import call_batch, move_batch
from .calibration.probe import StatisticsRegistry
from .graph.schedule import QuantizationSchedule
from .pipeline import QuantizationPipeline
from .plan import QuantizationPlan
from .recipe_config import QuantizationRecipeConfig
from .t2n.model_io import load_quantized_model, save_quantized_model


def _fingerprint(value: Any, digest: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value):
            digest.update(str(key).encode())
            _fingerprint(value[key], digest)
    elif isinstance(value, list | tuple):
        digest.update(str((type(value).__name__, len(value))).encode())
        for item in value:
            _fingerprint(item, digest)
    else:
        digest.update(repr(value).encode())


@dataclass
class BlockCalibrationResult:
    """Adapted block and aligned CPU outputs before/after quantization.

    ``reference_outputs`` always come from the original float block applied to
    float-stream inputs. ``student_outputs`` use quantized upstream inputs and
    the adapted block. A resumed result restores both streams and the model.
    """

    name: str
    model: nn.Module
    reference_outputs: list[Any]
    student_outputs: list[Any]
    kept_float: dict[str, str]
    resumed: bool = False
    schedule: QuantizationSchedule | None = None


class BlockQuantizationPipeline:
    """Execute recipes one float block at a time, with optional teacher observations.

    ``input_source='float'`` collects each block's statistics on the original
    float stream, equivalent to a full float-model collection for a deterministic
    partition. ``'sequential'`` collects on the adapted upstream stream. The
    estimator's requests, including paired collectors, determine the objective.

    ``inference_model`` may supply an inference view of an adapted block, for
    example a copy with packed weights materialized for GPU calibration. It must
    not mutate the original block. Its semantics must be included in ``run_id``
    when checkpointing. Block inputs/outputs are CPU tensor trees between calls.

    ``target_paths`` enables dependency stages within each block: collect and
    adapt independent targets together, then recollect for dependent targets.
    The trace requires a functional eval forward; teacher weights stay frozen.
    """

    def __init__(
        self,
        config: QuantizationRecipeConfig,
        *,
        input_source: Literal["float", "sequential"] = "sequential",
        forward_device: str | torch.device = "cpu",
        estimation_device: str | torch.device = "cpu",
        checkpoint_dir: str | Path | None = None,
        run_id: str | None = None,
        inference_model: Callable[[nn.Module], nn.Module] | None = None,
        target_paths: tuple[str, ...] | None = None,
    ) -> None:
        if input_source not in {"float", "sequential"}:
            raise ValueError("input_source must be float or sequential")
        if config.preprocessing:
            raise ValueError("Apply whole-model preprocessing before block calibration")
        if config.kernel_lowering is not None:
            raise ValueError(
                "Set kernel_lowering=None; lower kernels after block calibration"
            )
        if checkpoint_dir is not None and not run_id:
            raise ValueError(
                "Checkpointing requires run_id identifying the model/data/forward contract"
            )
        if target_paths is not None and input_source != "sequential":
            raise ValueError("Dependency stages require input_source=sequential")
        self.target_paths = tuple(target_paths) if target_paths is not None else None
        self.config = config
        self.input_source = input_source
        self.forward_device = torch.device(forward_device)
        self.estimation_device = torch.device(estimation_device)
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else None
        )
        self.run_id = run_id
        self.inference_model = inference_model

    @torch.no_grad()
    def forward(self, model: nn.Module, batches: list[Any]) -> list[Any]:
        """Run batches on the forward device and return detached CPU outputs."""
        model.to(self.forward_device).eval()
        outputs = [
            move_batch(call_batch(model, move_batch(batch, self.forward_device)), "cpu")
            for batch in batches
        ]
        model.cpu()
        return outputs

    def _checkpoint(
        self, name: str, model: nn.Module, student: list[Any], reference: list[Any]
    ) -> tuple[Path | None, str]:
        if self.checkpoint_dir is None:
            return None, ""
        digest = hashlib.sha256()
        _fingerprint(
            {
                "schema": 1,
                **(
                    {"target_paths": self.target_paths}
                    if self.target_paths is not None
                    else {}
                ),
                "run_id": self.run_id,
                "name": name,
                "recipe": self.config.model_dump_json(),
                "input_source": self.input_source,
                "forward_device": str(self.forward_device),
                "estimation_device": str(self.estimation_device),
                "torch": torch.__version__,
                "model": model.state_dict(),
                "student": student,
                "reference": reference,
            },
            digest,
        )
        path = self.checkpoint_dir / hashlib.sha256(name.encode()).hexdigest()[:20]
        return path, digest.hexdigest()

    def _stage_pairs(self, model: nn.Module, stage: tuple[str, ...] | None) -> bool:
        """Whether the recipe would request paired statistics for *stage*.

        Probes a separately built estimator, so the one that later runs the
        stage is prepared exactly as often as in an unprobed run. The probed
        targets are still float here, as they will be when their stage runs.
        """
        estimator = self.config.build().param_estimator
        with (
            estimator.select_targets(set(stage)) if stage is not None else nullcontext()
        ):
            estimator.prepare(model)
            registry = StatisticsRegistry()
            registry.register_requests(estimator.declare_statistics())
            return registry.requires_pairing()

    def run_block(
        self,
        name: str,
        model: nn.Module,
        student_data: Iterable,
        reference_data: Iterable,
    ) -> BlockCalibrationResult:
        """Calibrate and adapt a fresh float block; return both propagated streams.

        Batches must be aligned, nonempty, and accepted by the block's forward.
        Statistics use the forward device; estimation and packing use the
        estimation device. Checkpoints validate the recipe, inputs and float
        weights before loading, and commit metadata only after saving all state.
        """
        student = [move_batch(batch, "cpu") for batch in student_data]
        reference = [move_batch(batch, "cpu") for batch in reference_data]
        if not student or len(student) != len(reference):
            raise ValueError(
                "Nonempty, aligned student and reference batches are required"
            )
        path, fingerprint = self._checkpoint(name, model, student, reference)
        if path is not None and (path / "complete.json").exists():
            metadata = json.loads((path / "complete.json").read_text())
            if metadata["fingerprint"] != fingerprint:
                raise ValueError(
                    f"Checkpoint mismatch for block {name}; use a fresh checkpoint directory"
                )
            model.cpu()
            load_quantized_model(
                model, path / "model.safetensors", kernel_lowering=None
            )
            model.eval()
            streams = torch.load(path / "streams.pt", weights_only=True)
            return BlockCalibrationResult(
                name,
                model,
                streams["reference"],
                streams["student"],
                metadata["kept_float"],
                True,
                QuantizationSchedule(
                    tuple(tuple(stage) for stage in metadata["schedule"])
                )
                if metadata.get("schedule") is not None
                else None,
            )

        reference_outputs = self.forward(model, reference)
        source = reference if self.input_source == "float" else student
        schedule = None
        # Without target_paths the whole block is one stage that selects
        # every estimator target, exactly like a whole-model pipeline run.
        stages: tuple[tuple[str, ...] | None, ...] = (None,)
        if self.target_paths is not None:
            model.to(self.forward_device).eval()
            schedule = QuantizationSchedule.trace(
                model,
                (move_batch(batch, self.forward_device) for batch in source),
                self.target_paths,
            )
            model.cpu()
            stages = schedule.stages
        # The first stage collects before any adaptation, so the observed block
        # is its own float teacher. Later stages see adapted weights and need a
        # frozen copy, taken now (before stage one adapts anything) but only if
        # one of them will pair; it stays in host memory and reaches the
        # forward device only for the stages that do.
        teacher = (
            copy.deepcopy(model)
            if any(self._stage_pairs(model, stage) for stage in stages[1:])
            else None
        )

        def place_teacher() -> nn.Module:
            assert teacher is not None
            return teacher.to(self.forward_device).eval()

        plan = QuantizationPlan()
        try:
            for index, stage in enumerate(stages):
                observed = (
                    self.inference_model(model) if self.inference_model else model
                )
                observed.to(self.forward_device).eval()
                pipeline = QuantizationPipeline(self.config.build())
                pipeline.run(
                    model,
                    (move_batch(batch, self.forward_device) for batch in source),
                    reference_data=(
                        move_batch(batch, self.forward_device) for batch in reference
                    ),
                    reference_model=place_teacher if index else None,
                    estimation_device=self.estimation_device,
                    collection_model=observed,
                    target_paths=set(stage) if stage is not None else None,
                )
                plan.kept_float.update(pipeline.last_plan.kept_float)
                observed.cpu()
                model.cpu()
                if teacher is not None:
                    teacher.cpu()
        finally:
            if teacher is not None:
                teacher.cpu()
        model.cpu()
        inference = (
            self.inference_model(model) if self.inference_model is not None else model
        )
        student_outputs = self.forward(inference, student)
        result = BlockCalibrationResult(
            name,
            model,
            reference_outputs,
            student_outputs,
            dict(plan.kept_float),
            schedule=schedule,
        )
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
            save_quantized_model(
                model, path / "model.safetensors", self.config.model_dump_json()
            )
            torch.save(
                {"reference": reference_outputs, "student": student_outputs},
                path / "streams.pt",
            )
            temporary = path / "complete.json.tmp"
            temporary.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "kept_float": result.kept_float,
                        "schedule": schedule.stages if schedule else None,
                    }
                )
            )
            temporary.replace(path / "complete.json")
        return result

    def run(
        self,
        blocks: Iterable[tuple[str, nn.Module]],
        calibration_data: Iterable,
        *,
        propagate: Callable[[Any, Any], Any] | None = None,
    ) -> Iterator[BlockCalibrationResult]:
        """Yield adapted blocks while advancing aligned float/student activations.

        Blocks may be loaded lazily. ``propagate(previous_batch, output)`` forms
        the next block's batch; the default uses the output directly, so a dict
        output is passed to the next block as keyword arguments. Outputs are
        plain CPU tensor trees (see :func:`move_batch`): a ``ModelOutput`` is
        a ``dict`` here, fresh or resumed, so ``propagate`` should read
        ``output["last_hidden_state"]`` rather than the attribute. Retain only
        the results needed by the caller to preserve bounded memory.
        """
        reference = [move_batch(batch, "cpu") for batch in calibration_data]
        student = reference
        seen = set()
        # Two records, because run_block moves each block to the forward
        # device and back, which gives every parameter fresh storage: identity
        # catches a Parameter object reused by a later block, and the storage
        # snapshot catches distinct Parameters viewing the same memory (those
        # keep the original storage alive, so its weakref stays valid).
        # Keyed by id with a liveness check: Tensor.__eq__ is elementwise, so
        # set membership cannot compare Parameters by identity.
        visited: dict[int, weakref.ref] = {}
        parameters = {}
        for name, block in blocks:
            if name in seen:
                raise ValueError(f"Duplicate block name: {name}")
            seen.add(name)
            current_parameters = {}
            current_visited = {}
            for parameter in block.parameters():
                if parameter.numel() == 0:
                    continue
                storage = (
                    str(parameter.device),
                    parameter.untyped_storage().data_ptr(),
                )
                previous = parameters.get(storage)
                reused = visited.get(id(parameter))
                if (reused is not None and reused() is parameter) or (
                    previous is not None and previous() is not None
                ):
                    raise ValueError(
                        "Blocks must not share parameter storage across boundaries"
                    )
                current_parameters[storage] = weakref.ref(parameter.untyped_storage())
                current_visited[id(parameter)] = weakref.ref(parameter)
            parameters.update(current_parameters)
            visited.update(current_visited)
            result = self.run_block(name, block, student, reference)
            student = [
                propagate(batch, output) if propagate else output
                for batch, output in zip(student, result.student_outputs, strict=True)
            ]
            reference = [
                propagate(batch, output) if propagate else output
                for batch, output in zip(
                    reference, result.reference_outputs, strict=True
                )
            ]
            yield result
