# Bonsai block calibration comparison

This experiment compares three Q4_0 calibration objectives on the Bonsai text
encoder. Each uses the same source weights, calibration prompts, quantization
format and held-out evaluation prompts.

| Method | Statistics inputs | Reconstruction target |
| --- | --- | --- |
| Float statistics | Original float activations | Original linear applied to those inputs |
| Sequential | Quantized-upstream activations | Original linear applied to those inputs |
| Teacher | Quantized-upstream activations | Original linear on the original float stream |

Teacher reconstruction fits a ridge-regularized correction to each linear before
GPTQ. Its damping is fixed at 0.01 times the mean input Hessian diagonal. Biases
remain unchanged. All seven linear weights within a decoder block are calibrated
before adapting that block: this is not nonlinear whole-block optimization.

## Results

| Method | Relative L2: all tokens | Relative L2: non-padding | RMSE: all tokens | RMSE: non-padding |
| --- | ---: | ---: | ---: | ---: |
| Float statistics + GPTQ | 0.892549% | 0.541264% | 0.374753 | 0.264343 |
| Sequential GPTQ | 0.882883% | 0.528381% | 0.370695 | 0.258051 |
| Teacher linear reconstruction + GPTQ | 2.643917% | 2.082812% | 1.110097 | 1.017205 |

Sequential GPTQ reduces relative L2 error versus float statistics by
1.08% over all tokens and 2.38% over non-padding tokens.
Its all-token error is lower on 17/32 individual held-out prompts: this is a
small aggregate gain, not a consistent per-prompt improvement.
Teacher linear reconstruction is worse here: its error is 2.99 times
sequential GPTQ over all tokens and 3.94 times over non-padding tokens.
The measured result supports retaining sequential self-reconstruction for this
Bonsai calibration; optimizing a local teacher objective does not ensure a better
final encoder. This experiment does not evaluate nonlinear whole-block fitting.

Each mode quantizes 189 linear weights, approximately 2.725 billion parameters,
at 4.5 bits per quantized weight. All saved codes are valid Q4_0 integers and
all group scales are exactly representable in f16. The sequential runner's
integer codes and scales match the previously published calibration exactly
across all 189 weights.

[Machine-readable results](bonsai-block-calibration.json) include all 27 block
measurements, per-prompt errors, environment versions and the prompt-selection
hash. The numerical calibration implementation is from commit
`d1c0354366694a0d6f10b96d91d85a59ff379cca`.

## Reproduction

Run from the NeuroPress checkout:

```sh
python -m benchmarks.bonsai_block_calibration \
  --snapshot MODEL_SNAPSHOT --prompts prompts.json --out OUTPUT_DIRECTORY
```

The output directory contains recipes, packed codes/scales, restart checkpoints,
per-block measurements and final measurements.

- Model: `prism-ml/bonsai-image-ternary-4B-unpacked`, revision
  `30bbc032dd26939bff95c3d523477ad5065c3e1a`.
- Dataset: `Gustavosta/Stable-Diffusion-Prompts`, revision
  `d816d4a05cb89bde39dd99284c459801e1e7e69a`.
- Selection: the published Bonsai GPTQ script's 256 calibration prompts and 32
  disjoint held-out prompts, seed 3309, complete templated prompts up to 64 tokens.
- Quantization: symmetric four-bit weights in groups of 32, Q4_0 integer codes
  and f16 scales. Evaluation explicitly materializes these final weights in f32.
- Forward device: Apple MPS. Estimation: CPU, eight Torch threads. Batch size: 8.
- Output: concatenated hidden states from decoder blocks 9, 18 and 27.
- Evaluation: relative L2 and RMSE against the float encoder, separately for all
  positions and non-padding positions. No held-out prompts are used for fitting
  or choosing parameters.

These are encoder reconstruction measurements for one model and prompt sample.
They do not measure generated-image quality or compare optimized runtime speed.

The prompt selection is produced by `prepare()` in the
[published calibration script](https://huggingface.co/tract-nnef/bonsai-image-ternary-4B/blob/70610a5817027d0e097e18ff95bb470d21ebf845/export/calibrate_gptq.py).
Its `prompts.json` SHA-256 is
`0e9e92053792efa32fbea187aeb5a338069756f221bbe97dd81510e74ee1b7b0`.
The environment uses Python 3.13.7, Torch 2.11.0, torch-to-nnef 0.25.0,
Transformers 5.6.2, safetensors 0.8.0 and Pydantic 2.14.3 on macOS arm64.
NeuroPress is the PR source checkout, not the installed 0.2.0 release.
