"""Fine-tune VQ codebooks after quantization, with the indices frozen.

A VQ weight is ``W = C[idx]`` (then column-unpermuted), so with ``idx`` fixed
the loss is differentiable in the codebook ``C`` alone and the gradient of the
gather is an ``index_add`` into ``C``: only codebook-sized gradients exist, and
the trainable state is tiny (a 4096 x 4 codebook per group).  Training the
codebooks end to end against the unquantized model corrects the error that
per-layer, Hessian-weighted k-means cannot see: how one layer's quantization
error propagates through the following ones.

Usage::

    groups = prepare_vq_finetune(student)          # swaps VQ linears in place
    finetune_codebooks(groups, student_fn, teacher_fn, batches, lr=1e-4)
    write_back_codebooks(student, groups)          # restores QTensor linears

The checkpoint format does not change: only centroid values move, so the
result saves and loads through :func:`~neuropress.t2n.save_quantized_model`
and runs on every existing VQ kernel.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

LOGGER = logging.getLogger(__name__)


class TrainableVQLinear(nn.Module):
    """``nn.Linear`` whose weight is ``codebook[idx]``, with ``idx`` frozen.

    Holds the unpacked indices (int32) and the column permutation; the
    codebook is a shared :class:`nn.Parameter` owned by a :class:`CodebookGroup`.
    """

    def __init__(
        self,
        original: nn.Linear,
        codebook: nn.Parameter,
        indices: torch.Tensor,
        column_perm: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.original = [original]  # list: keep it out of the module tree
        self.out_features = original.out_features
        self.in_features = original.in_features
        self.codebook = codebook
        self.register_buffer("indices", indices, persistent=False)
        self.register_buffer("column_perm", column_perm, persistent=False)
        self.bias = original.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear map with ``W = codebook[indices]`` in permuted layout."""
        if self.column_perm is not None:
            x = x[..., self.column_perm]
        w = self.codebook[self.indices].reshape(self.out_features, self.in_features)
        return F.linear(x, w.to(x.dtype), self.bias)


@dataclass
class CodebookGroup:
    """One trainable codebook and every layer that shares it."""

    codebook: nn.Parameter
    schemes: list[Any] = field(default_factory=list)
    names: list[str] = field(default_factory=list)


def _is_vq_linear(mod: nn.Module) -> bool:
    w = getattr(mod, "weight", None)
    return isinstance(mod, nn.Linear) and type(w).__name__ == "QTensorTractVQ"


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, new)


def prepare_vq_finetune(
    model: nn.Module, device: torch.device | str | None = None
) -> list[CodebookGroup]:
    """Replace every VQ ``nn.Linear`` in *model* with a :class:`TrainableVQLinear`.

    Layers whose schemes share one centroid tensor (shared codebooks, as
    restored by the checkpoint loader) share one fp32 trainable codebook.
    Every other parameter of *model* is frozen.

    Args:
        model: Model whose VQ layers are ``nn.Linear`` with a ``QTensorTractVQ``
            weight (load a checkpoint with ``kernel_lowering=None``).
        device: Where the indices and codebooks live; defaults to each
            weight's device.

    Returns:
        The codebook groups, in first-seen order.

    Raises:
        ValueError: If a layer uses a tiered or zero-escape index format, or
            if *model* has no VQ layers.
    """
    from ..transforms.vq_packing import IndexPacking

    for p in model.parameters():
        p.requires_grad_(False)
    groups: dict[int, CodebookGroup] = {}
    targets = [(n, m) for n, m in model.named_modules() if _is_vq_linear(m)]
    if not targets:
        raise ValueError("prepare_vq_finetune: model has no VQ linear layers")
    for name, mod in targets:
        q = mod.weight
        scheme = q.qscheme
        if scheme.use_tiered or scheme.zero_escape or scheme.tier_bits is not None:
            raise ValueError(f"{name}: only flat index packing is supported")
        dev = torch.device(device) if device is not None else q.u8_blob.device
        key = id(scheme.centroids)
        if key not in groups:
            codebook = nn.Parameter(
                scheme.centroids.detach().to(dev, torch.float32).clone()
            )
            groups[key] = CodebookGroup(codebook=codebook)
        group = groups[key]
        n_vectors = mod.out_features * mod.in_features // scheme.vector_len
        idx = IndexPacking.unpack(
            q.u8_blob.to(dev), scheme.is_packed_indices, n_vectors
        )
        perm = scheme.column_permutation
        new = TrainableVQLinear(
            mod,
            group.codebook,
            idx.to(torch.int32),
            perm.to(dev) if perm is not None else None,
        )
        _set_submodule(model, name, new)
        group.schemes.append(scheme)
        group.names.append(name)
    LOGGER.info(
        "vq_finetune: %d VQ layers, %d trainable codebooks (%d parameters)",
        len(targets),
        len(groups),
        sum(g.codebook.numel() for g in groups.values()),
    )
    return list(groups.values())


def write_back_codebooks(model: nn.Module, groups: list[CodebookGroup]) -> None:
    """Copy trained codebooks into the VQ schemes and restore the original layers.

    The schemes' centroid tensors are updated in place (fp16), so layers that
    share a codebook keep sharing it.
    """
    for group in groups:
        new = group.codebook.detach()
        seen: set[int] = set()
        for scheme in group.schemes:
            if id(scheme.centroids) in seen:
                continue
            seen.add(id(scheme.centroids))
            scheme.centroids.copy_(
                new.to(scheme.centroids.device, scheme.centroids.dtype)
            )
    for name, mod in list(model.named_modules()):
        if isinstance(mod, TrainableVQLinear):
            _set_submodule(model, name, mod.original[0])


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Mean per-token KL(teacher || student) over the last dimension, in fp32."""
    s = F.log_softmax(student_logits.float(), dim=-1)
    t = F.log_softmax(teacher_logits.float(), dim=-1)
    kl = F.kl_div(s, t, log_target=True, reduction="none").sum(-1)
    return kl.mean()


def finetune_codebooks(
    groups: list[CodebookGroup],
    student_fn: Callable[[Any], torch.Tensor],
    teacher_fn: Callable[[Any], torch.Tensor],
    batches: Iterable[Any],
    *,
    lr: float = 1e-4,
    epochs: int = 1,
    warmup_frac: float = 0.05,
    log_every: int = 10,
) -> list[float]:
    """Train the codebooks so the student's logits match the teacher's.

    Args:
        groups: From :func:`prepare_vq_finetune`.
        student_fn: Batch -> student logits (differentiable in the codebooks).
        teacher_fn: Batch -> teacher logits (run under ``no_grad``).
        batches: Re-iterable collection of batches (a list); one optimizer
            step per batch per epoch.
        lr: Peak AdamW learning rate (no weight decay: centroids are values,
            not overparameterized weights).
        epochs: Passes over *batches*.
        warmup_frac: Linear warm-up fraction, then cosine decay to zero.
        log_every: Log the running loss every this many steps.

    Returns:
        The loss of every step.
    """
    batches = list(batches)
    params = [g.codebook for g in groups]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total = max(len(batches) * epochs, 1)
    warmup = max(int(total * warmup_frac), 1)

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    losses: list[float] = []
    step = 0
    for _ in range(epochs):
        for batch in batches:
            with torch.no_grad():
                target = teacher_fn(batch)
            loss = kd_loss(student_fn(batch), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % log_every == 0 or step == total:
                recent = losses[-log_every:]
                LOGGER.info(
                    "vq_finetune: step %d/%d  kd %.5f  lr %.2e",
                    step,
                    total,
                    sum(recent) / len(recent),
                    sched.get_last_lr()[0],
                )
    for p in params:
        p.requires_grad_(False)
    return losses
