"""Hadamard incoherence processing for weight quantization.

Random Hadamard rotation makes weight matrices approximately i.i.d. Gaussian,
which suppresses outlier channels and conditions the Hessian.  This is a
prerequisite for high vector-length VQ: lattice and trellis codebooks assume a
Gaussian source and degrade badly on anisotropic, outlier-heavy weights.

The transform applied is ``H · D``, where ``D`` is a random ±1 diagonal and
``H`` is the orthonormal Hadamard matrix.  Both are orthogonal, so the pair is
norm-preserving and exactly invertible.
"""

import torch
from torch import nn


def _legendre_symbol(a: int, q: int) -> int:
    """Quadratic residue character of *a* modulo prime *q*."""
    a %= q
    if a == 0:
        return 0
    return 1 if pow(a, (q - 1) // 2, q) == 1 else -1


def _paley_hadamard(q: int) -> torch.Tensor:
    """Paley type-I Hadamard matrix of order ``q + 1``.

    Valid when *q* is a prime with ``q % 4 == 3``, which makes the Jacobsthal
    matrix skew-symmetric and hence ``H = I + S`` Hadamard.
    """
    n = q + 1
    jacobsthal = torch.tensor(
        [[_legendre_symbol(i - j, q) for j in range(q)] for i in range(q)],
        dtype=torch.float64,
    )
    s = torch.zeros(n, n, dtype=torch.float64)
    s[0, 1:] = 1.0
    s[1:, 0] = -1.0
    s[1:, 1:] = jacobsthal
    return torch.eye(n, dtype=torch.float64) + s


def hadamard_matrix(n: int) -> torch.Tensor:
    """Return an ``n x n`` Hadamard matrix with ±1 entries (not normalized).

    Supports powers of two (Sylvester doubling) and order 12 (Paley), which
    together cover every dimension our target architectures need.

    Raises:
        ValueError: If no construction is available for *n*.
    """
    if n == 12:
        return _paley_hadamard(11)
    if n >= 1 and (n & (n - 1)) == 0:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        return h
    if n % 12 == 0 and (n // 12) >= 1 and ((n // 12) & (n // 12 - 1)) == 0:
        # Kronecker order must match fast_hadamard_transform's factorization.
        return torch.kron(_paley_hadamard(11), hadamard_matrix(n // 12))
    raise ValueError(f"No Hadamard construction available for order {n}.")


def random_hadamard_matrix(n: int, seed: int) -> torch.Tensor:
    """Return the orthonormal random Hadamard rotation ``H · D`` of order *n*.

    ``D`` is a ±1 diagonal drawn from *seed*, so the whole matrix is recoverable
    from a single integer and never needs to be serialized.  Both factors are
    orthogonal, making the product norm-preserving and exactly invertible via
    its transpose.

    Args:
        n: Transform order; must have a Hadamard construction.
        seed: Integer seed determining the sign flips.

    Returns:
        An ``n x n`` orthonormal matrix in float64.
    """
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int64)
    diag = (signs * 2 - 1).to(torch.float64)
    return (hadamard_matrix(n) / n**0.5) * diag


def rotation_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the dtype a runtime rotation should be computed in.

    An orthogonal transform only cancels its folded inverse if both are
    evaluated with enough precision.  Weight folds run in float64, so applying
    the matching runtime transform in the model's storage dtype breaks the
    cancellation.  In bf16 the residual relative logit error is orders of
    magnitude larger than in float32 - large enough to destroy model quality
    outright.  Anything below float32 is therefore promoted.
    """
    return dtype if dtype in (torch.float32, torch.float64) else torch.float32


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _fwht_power_of_two(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal fast Walsh-Hadamard transform along the last dimension."""
    n = x.shape[-1]
    orig_shape = x.shape
    y = x.reshape(-1, n).clone()

    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack([a + b, a - b], dim=2).reshape(-1, n)
        h *= 2

    return (y / n**0.5).reshape(orig_shape)


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply the orthonormal Hadamard transform along the last dimension.

    Handles powers of two directly via the butterfly, and sizes of the form
    ``12 * 2^k`` via the Kronecker factorization ``H_12 ⊗ H_2^k``.  That covers
    every dimension our target architectures use: Gemma-4-E2B's hidden size
    1536 = 12 x 128 and intermediate size 6144 = 12 x 512 are both non-powers
    of two, while head_dim 256 is a plain power of two.

    Args:
        x: Tensor whose last dimension is a supported transform size.

    Returns:
        Tensor of the same shape, transformed along the last dimension.

    Raises:
        ValueError: If the last dimension has no available construction.
    """
    n = x.shape[-1]
    if _is_power_of_two(n):
        return _fwht_power_of_two(x)

    if n % 12 == 0 and _is_power_of_two(n // 12):
        m = n // 12
        orig_shape = x.shape
        y = _fwht_power_of_two(x.reshape(-1, 12, m))
        h12 = hadamard_matrix(12).to(device=y.device, dtype=y.dtype) / 12**0.5
        y = torch.einsum("ki,bij->bkj", h12, y)
        return y.reshape(orig_shape)

    raise ValueError(
        f"Unsupported Hadamard transform size {n}: expected a power of two or 12 * 2^k."
    )


def rotate_attention_vo(
    v_proj: nn.Linear,
    o_proj: nn.Linear,
    head_dim: int,
    seed: int,
) -> None:
    """Rotate the value/output attention path in place, exactly preserving output.

    Only sequence-axis mixing (the attention weights) sits between ``v_proj``
    and ``o_proj``, and that never mixes the ``head_dim`` axis.  So applying an
    orthogonal ``R`` per head on v_proj's output and ``Rᵀ`` per head on o_proj's
    input cancels exactly:  ``W_o Rᵀ · R W_v = W_o W_v``.

    This is the only rotation that folds with *zero* runtime cost in a
    sandwich-norm architecture such as Gemma-4, where ``post_attention_layernorm``
    and ``post_feedforward_layernorm`` sit after their sublayers and block the
    usual residual-stream rotation.

    Grouped-query attention is handled: ``v_proj`` may have fewer heads than
    ``o_proj``, since the same per-head ``R`` is applied to every head on both
    sides.  That also makes the transform safe under KV sharing, provided every
    layer sharing a projection uses the same seed.

    Args:
        v_proj: Value projection, modified in place.
        o_proj: Output projection, modified in place.
        head_dim: Per-head dimension; must divide both projections and have a
            Hadamard construction.
        seed: Seed for the ±1 diagonal.

    Raises:
        ValueError: If *head_dim* does not divide the projection shapes.
    """
    if v_proj.weight.shape[0] % head_dim or o_proj.weight.shape[1] % head_dim:
        raise ValueError(
            f"head_dim={head_dim} must divide v_proj out_features "
            f"({v_proj.weight.shape[0]}) and o_proj in_features "
            f"({o_proj.weight.shape[1]})."
        )

    rot = random_hadamard_matrix(head_dim, seed).to(v_proj.weight.device)

    weight = v_proj.weight.data
    dtype = weight.dtype
    n_kv = weight.shape[0] // head_dim
    rotated = torch.einsum(
        "ab,kbi->kai", rot, weight.to(torch.float64).view(n_kv, head_dim, -1)
    )
    v_proj.weight.data = rotated.reshape(n_kv * head_dim, -1).to(dtype)

    if v_proj.bias is not None:
        bias = v_proj.bias.data
        rotated_bias = torch.einsum(
            "ab,kb->ka", rot, bias.to(torch.float64).view(n_kv, head_dim)
        )
        v_proj.bias.data = rotated_bias.reshape(-1).to(bias.dtype)

    # o_proj's bias is added after the rotated input is consumed, so it is
    # unaffected and deliberately left alone.
    out_weight = o_proj.weight.data
    out_dtype = out_weight.dtype
    n_q = out_weight.shape[1] // head_dim
    rotated_out = (
        out_weight.to(torch.float64).view(-1, n_q, head_dim)
        @ rot.to(out_weight.device).T
    )
    o_proj.weight.data = rotated_out.reshape(-1, n_q * head_dim).to(out_dtype)


def fold_rms_norm_into_linears(
    norm: nn.Module,
    linears: list[nn.Linear],
    plus_one: bool = True,
) -> None:
    """Move an RMSNorm's elementwise scale into the linears that consume it.

    A residual-stream rotation requires the norm to be a *pure* RMS
    normalization, because ``x / rms(x)`` commutes with an orthogonal ``R``
    (``||Rx|| == ||x||``) while an elementwise scale does not.  Folding the
    scale forward into every consumer leaves exactly that pure form.

    Args:
        norm: Module with a ``weight`` attribute, modified in place to become
            an identity scale.
        linears: Every layer consuming *norm*'s output.  Missing one silently
            corrupts the model, so callers must pass the complete set.
        plus_one: Whether the norm applies ``1 + weight`` (Qwen3.5, Gemma) as
            opposed to plain ``weight``.  Identity is stored as 0 under the
            ``+1`` convention and 1 otherwise.
    """
    scale = norm.weight.data.to(torch.float64)
    if plus_one:
        scale = 1.0 + scale

    for linear in linears:
        weight = linear.weight.data
        dtype = weight.dtype
        linear.weight.data = (weight.to(torch.float64) * scale.unsqueeze(0)).to(dtype)

    identity = 0.0 if plus_one else 1.0
    norm.weight.data = torch.full_like(norm.weight.data, identity)


def _rotate_linear_input(linear: nn.Linear, rot: torch.Tensor) -> None:
    """``W -> W Rᵀ`` so that ``linear(R x)`` reproduces the original output."""
    weight = linear.weight.data
    dtype = weight.dtype
    rot = rot.to(weight.device)
    linear.weight.data = (weight.to(torch.float64) @ rot.T).to(dtype)


def _rotate_linear_output(linear: nn.Linear, rot: torch.Tensor) -> None:
    """``W -> R W`` so the layer emits its output already in the rotated basis."""
    weight = linear.weight.data
    dtype = weight.dtype
    rot = rot.to(weight.device)
    linear.weight.data = (rot @ weight.to(torch.float64)).to(dtype)
    if linear.bias is not None:
        bias = linear.bias.data
        linear.bias.data = (rot @ bias.to(torch.float64)).to(bias.dtype)


def rotate_residual_stream(
    layers: list[dict],
    hidden_size: int,
    seed: int,
    plus_one_norms: bool = True,
) -> torch.Tensor:
    """Rotate a pre-norm model's residual stream in place, preserving its output.

    Available on architectures whose sublayers end in a plain ``Linear`` writing
    to the residual, and whose norms are all *pre*-norms - Qwen3.5 qualifies,
    Gemma-3/4 do not (their sandwich norms sit after each sublayer, where an
    elementwise scale can no longer be folded forward).

    Each layer entry is a dict with keys ``input_norm``, ``input_consumers``,
    ``attn_output``, ``post_norm``, ``post_consumers``, ``mlp_output``.
    ``input_consumers`` must list *every* projection reading the norm's output
    (for Qwen3.5's GatedDeltaNet blocks that is ``in_proj_qkv/z/b/a``, not just
    the qkv projection); omitting one silently corrupts the model.

    The embedding table is deliberately **not** rotated.  Qwen3.5 ties it to
    ``lm_head``, and the two roles demand different corrections: the embedding
    role wants ``W Rᵀ`` to emit rotated vectors, while the output projection
    wants ``W`` unchanged once its input has been un-rotated.  No single
    transform satisfies both, and untying would cost a second
    ``[vocab, hidden]`` tensor.  Leaving it alone loses nothing, since the
    embedding is not a quantization target in this recipe.

    The caller therefore brackets the stack with two cheap ops on the hidden
    state: apply ``R`` right after the embedding, and ``Rᵀ`` right before the
    final norm.  In row-vector convention those are ``x @ R.T`` and ``x @ R``
    respectively.  That is two FWHTs per forward pass in total, not per layer.

    Args:
        layers: Per-layer module references, described above.
        hidden_size: Residual stream width; needs a Hadamard construction.
        seed: Seed for the rotation.
        plus_one_norms: Whether norms use the ``1 + weight`` convention.

    Returns:
        The rotation matrix ``R``, for the caller's bracketing ops.
    """
    rot = random_hadamard_matrix(hidden_size, seed)

    for layer in layers:
        fold_rms_norm_into_linears(
            layer["input_norm"], layer["input_consumers"], plus_one_norms
        )
        for consumer in layer["input_consumers"]:
            _rotate_linear_input(consumer, rot)
        _rotate_linear_output(layer["attn_output"], rot)

        fold_rms_norm_into_linears(
            layer["post_norm"], layer["post_consumers"], plus_one_norms
        )
        for consumer in layer["post_consumers"]:
            _rotate_linear_input(consumer, rot)
        _rotate_linear_output(layer["mlp_output"], rot)

    return rot


class HadamardInputLinear(nn.Module):
    """A ``Linear`` whose input is Hadamard-rotated online at every forward.

    Used where a rotation cannot be folded backwards.  In an MLP the
    intermediate passes through ``silu(gate) * up`` - elementwise and
    nonlinear - so rotating it cannot be absorbed into ``gate_proj`` or
    ``up_proj`` and must happen at runtime instead.

    The matching ``Rᵀ`` is folded into ``linear``'s weight, so this wrapper
    reproduces the original layer's output exactly while the stored weight is
    far more incoherent.

    Cost is one FWHT per forward, ``O(n log n)``, with no stored matrix: the
    ±1 diagonal is a buffer and the Hadamard factor is computed by butterfly.

    Installing this wrapper renames the quantization target from
    ``...mlp.down_proj`` to ``...mlp.down_proj.linear``.  That is intentional
    and load-bearing in two ways: the statistics collector then hooks the inner
    layer and so estimates the Hessian from the *rotated* activation, matching
    the basis the rotated weight lives in; and the adapter replaces the inner
    layer, leaving this wrapper in place rather than discarding the rotation.
    A plain forward-pre-hook on ``down_proj`` would achieve neither, since
    module replacement drops hooks bound to the old instance.
    """

    def __init__(self, linear: nn.Linear, signs: torch.Tensor) -> None:
        super().__init__()
        self.linear = linear
        self.register_buffer("signs", signs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the signed Hadamard rotation to *x*, then the wrapped linear.

        The rotation runs in :func:`rotation_compute_dtype` so it cancels the
        folded inverse; the result is cast back to the input dtype.

        Args:
            x: Input activations.

        Returns:
            Output of the wrapped linear, in the input dtype.
        """
        out_dtype = x.dtype
        compute_dtype = rotation_compute_dtype(out_dtype)
        signs = self.signs.to(device=x.device, dtype=compute_dtype)
        rotated = fast_hadamard_transform(x.to(compute_dtype) * signs)
        return self.linear(rotated.to(out_dtype))


def wrap_linear_with_online_hadamard(
    linear: nn.Linear,
    seed: int,
) -> HadamardInputLinear:
    """Fold ``Rᵀ`` into *linear* and return it wrapped in the online rotation.

    ``R = H·D`` is applied to the incoming activation at runtime while the
    weight absorbs ``Rᵀ``, leaving the layer's output unchanged.  Note the
    factor order: ``R x = H (d ⊙ x)``, so the wrapper applies the sign flip
    *before* the transform.

    Args:
        linear: Layer to rotate, modified in place.
        seed: Seed for the ±1 diagonal, matching the folded rotation.

    Returns:
        A module to install in place of *linear*.
    """
    in_features = linear.weight.shape[1]
    generator = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (in_features,), generator=generator, dtype=torch.int64)
    signs = (signs * 2 - 1).to(torch.float64).to(linear.weight.device)

    # W Rᵀ = W D Hᵀ = fwht(W ⊙ d), which is O(n log n) per row rather than the
    # O(n²) of materializing R.  At Qwen3.5's 6144 that matters across 24 layers.
    weight = linear.weight.data
    dtype = weight.dtype
    linear.weight.data = fast_hadamard_transform(weight.to(torch.float64) * signs).to(
        dtype
    )

    return HadamardInputLinear(linear, signs)


class UnrotateInput(nn.Module):
    """Wraps a module, un-rotating its input back to the original basis.

    Used on the final norm, whose scale cannot be folded into a tied
    ``lm_head``.  This is a *module wrapper* rather than a forward hook
    deliberately: kernel lowering replaces every RMSNorm instance via
    ``setattr``, and hooks bind to the instance, so a hook here is silently
    destroyed.  Wrapping instead leaves the inner module replaceable while the
    un-rotation survives - the same reason ``HadamardInputLinear`` wraps rather
    than hooks ``down_proj``.
    """

    def __init__(self, inner: nn.Module, rot: torch.Tensor) -> None:
        super().__init__()
        self.inner = inner
        self.register_buffer("rot", rot)

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object):
        """Un-rotate *x* back to the canonical basis, then call the inner module.

        Args:
            x: Activations in the rotated residual basis.
            *args: Forwarded positionally to the inner module.
            **kwargs: Forwarded by keyword to the inner module.

        Returns:
            Whatever the inner module returns.
        """
        out_dtype = x.dtype
        compute_dtype = rotation_compute_dtype(out_dtype)
        rot = self.rot.to(device=x.device, dtype=compute_dtype)
        return self.inner((x.to(compute_dtype) @ rot).to(out_dtype), *args, **kwargs)
