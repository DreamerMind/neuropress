"""lm-evaluation-harness integration for neuropress.

Provides :func:`evaluate` and :func:`compare` for running lm-eval tasks on
float and quantized models, plus :class:`EvalComparison` for inspecting the
results side-by-side.

A command-line interface is available via :mod:`.llm_cli` (entry point:
``neuropress-eval``)::

    neuropress-eval eval meta-llama/Llama-2-7b-hf --tasks hellaswag
    neuropress-eval compare meta-llama/Llama-2-7b-hf
        --tasks hellaswag --compression open_text_c4_gptq_q4_0
    neuropress-eval list

Requires the ``eval`` optional extra::

    pip install 'neuropress[eval]'
"""

from .harness import EvalComparison, compare, evaluate

__all__ = [
    "EvalComparison",
    "compare",
    "evaluate",
]
