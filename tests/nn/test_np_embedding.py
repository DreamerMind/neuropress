"""Tests for NPEmbedding.to_qtensor().

Regression coverage for the checkpoint-save crash found while re-running
the Qwen3.5-2B VPTQ index/frequency analysis (2026-07): NPEmbedding stores
its weight nibble-packed ([V, D // vpb]) for GPU-memory efficiency, but
to_qtensor() handed that packed blob straight to the generic
QScalePerGroupF16 serde, which assumes one quantized value per byte (the
format nn.Linear weights use -- Linear is never packed). Every downstream
shape ended up short by the packing factor, crashing
save_quantized_model() with a tensor-size mismatch.
"""

import torch

from neuropress.nn.np_embedding import NPEmbedding, _emb_dequant


def _pack_nibbles(values: torch.Tensor, nbits: int) -> torch.Tensor:
    """Pack an int tensor [V, D] with values in [0, 2**nbits) into
    [V, D // vpb] uint8, vpb values per byte -- the inverse of the
    unpacking done by _emb_dequant and _unpack_u8."""
    vpb = 8 // nbits
    V, D = values.shape
    assert D % vpb == 0
    grouped = values.to(torch.uint8).reshape(V, D // vpb, vpb)
    packed = torch.zeros(V, D // vpb, dtype=torch.uint8)
    for i in range(vpb):
        packed |= grouped[:, :, i] << (i * nbits)
    return packed


def _build_packed_embedding(V=4, D=32, nbits=4, group_size=8, backend="compile"):
    torch.manual_seed(0)
    n_groups = D // group_size
    quant_values = torch.randint(0, 2**nbits, (V, D))
    packed = _pack_nibbles(quant_values, nbits)
    scale = torch.rand(V * n_groups, dtype=torch.float16) + 0.1

    emb = NPEmbedding(
        u8_blob=packed,
        scale=scale,
        num_embeddings=V,
        embedding_dim=D,
        group_size=group_size,
        nbits=nbits,
        qtype="sym",
        backend=backend,
        qtensor_meta={
            "n_bits": str(nbits),
            "group_size": str(group_size),
            "dequant_to_dtype": "float16",
            "nnef_name": "",
        },
    )
    return emb, packed, scale


class TestToQTensor:
    def test_does_not_raise_on_packed_blob(self):
        emb, _, _ = _build_packed_embedding()
        # Previously raised RuntimeError: tensor size mismatch (packed blob
        # fed to an unpacking-unaware dequantizer).
        emb.to_qtensor()

    def test_roundtrip_matches_forward_dequant(self):
        V, D, nbits, group_size = 4, 32, 4, 8
        emb, packed, scale = _build_packed_embedding(V, D, nbits, group_size)

        q = emb.to_qtensor()
        actual = q.decompress().float()

        expected = _emb_dequant(
            torch.arange(V), packed, scale, nbits, group_size, None
        ).float()

        assert torch.allclose(actual, expected, atol=5e-2), (
            f"Max diff = {(actual - expected).abs().max():.4f}"
        )


class _EmbeddingWrapper:
    """Duck-typed stand-in for nn.Embedding: from_embedding() only reads
    .weight/.num_embeddings/.embedding_dim/.padding_idx, so a real quantized
    nn.Embedding (with all the QTensor-as-Parameter plumbing) isn't needed."""

    def __init__(self, weight, num_embeddings, embedding_dim, padding_idx=None):
        self.weight = weight
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx


class TestFromEmbeddingRelowerRoundtrip:
    """Regression coverage for the re-lowering crash found immediately after
    fixing to_qtensor(): save_quantized_model() unlowers (to_qtensor), saves,
    then re-lowers (from_embedding again) for the returned model to keep
    running fast. to_qtensor()'s output QTensor carries no u8_compressors
    (unlike the original, genuinely BitPacking4-compressed embedding), but
    from_embedding() assumed its input's u8_blob was always already in
    NPEmbedding's packed layout -- true for the original QTensor, false for
    the round-tripped one. Re-lowering then crashed with a shape mismatch
    inside _emb_dequant during the *next* eval, exactly like this test's
    second from_embedding() call reproduces.
    """

    def test_relower_after_to_qtensor_roundtrip(self):
        from torch_to_nnef.tensor.quant.base import (
            qscale_per_group_f16_min_max_calibration,
        )
        from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

        from neuropress.t2n.packing import BitPacking4

        torch.manual_seed(0)
        V, D, group_size = 4, 32, 8
        weight_f = torch.randn(V, D, dtype=torch.float16)
        q_scheme = qscale_per_group_f16_min_max_calibration(
            weight_f, n_bits=4, group_size=group_size
        )
        # Genuinely BitPacking4-compressed, like the real "q4_0_packed"
        # embedding weight transform produces.
        q1 = QTensorTractScaleOnly(
            weight_f,
            qscheme=q_scheme,
            dequant_to_dtype=weight_f.dtype,
            u8_compressors=[BitPacking4],
        )
        expected = q1.decompress().float()

        emb1 = NPEmbedding.from_embedding(
            _EmbeddingWrapper(q1, V, D), backend="compile"
        )
        assert emb1.u8_blob.shape == (V, D // 2)
        actual1 = emb1(torch.arange(V)).float()
        assert torch.allclose(actual1, expected, atol=5e-2)

        # First "unlower" (save) -- q2 carries no compressor.
        q2 = emb1.to_qtensor()
        assert q2.u8_compressors == []

        # Then "re-lower" -- previously crashed here (shape mismatch in
        # _emb_dequant at the *next* forward call) because from_embedding()
        # assumed q2.u8_blob was already packed like q1's was.
        emb2 = NPEmbedding.from_embedding(
            _EmbeddingWrapper(q2, V, D), backend="compile"
        )
        assert emb2.u8_blob.shape == (V, D // 2)
        actual2 = emb2(torch.arange(V)).float()
        assert torch.allclose(actual2, expected, atol=5e-2)
