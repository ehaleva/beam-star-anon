# BeamStar

BeamStar is a drop-in beam-search accelerator for vLLM that dynamically prunes the active beam set at each decoding step. It keeps only hypotheses whose per-token surprise rate stays within an adaptive margin of the model's capability frontier, reducing FLOPs, KV-cache traffic, and memory bandwidth without sacrificing generation quality.

It implements the algorithm described in the paper "BeamStar: Beam Pruning as Online Rate–Distortion Coding."

## How it works

Standard beam search keeps a fixed number of beams `k` at every step regardless of whether they are useful. BeamStar instead:

1. Estimates the **capability frontier** — the best surprise rate among current hypotheses.
2. Retains only beams within an **adaptive margin** `ε_t` of that frontier.
3. Accumulates the **distortion** (lost probability mass) against a user-specified budget `δ`, only re-admitting pruned beams if the budget would be exceeded.

The margin `ε_t` is wide early (exploration) and tightens as decoding progresses (exploitation), derived from concentration of measure.

## Installation

```bash
# Download the repository contents from the anonymized artifact link,
# then from the repository root:
pip install .
```

## Usage

BeamStar wraps vLLM's CLI. Pass the `--bs` flag to enable it:

```bash
beam-star-vllm serve <model-name> --bs [other vllm args]
```

### API

You can add the `"cum_logprob_threshold"` parameter to the `extra_body` (a.k.a. `vllm_xargs`) in chat completion requests to control the pruning aggressiveness:

```json
{
  "n": 128,
  "extra_body": {
    "cum_logprob_threshold": -10.0
  }
}
```

Lower (more negative) values are more permissive; higher (closer to zero) values prune more aggressively.

### Environment tuning

For token-range control of the exploration window, set the `EXPLORATION_TOKENS` environment variable before launching the server.

Preferred format:

```bash
export EXPLORATION_TOKENS="2,3,4,5"
```

This means the adaptive pruning rule is disabled for generated token indices `2` through `5` inclusive. 

## Components

- `beam_star/vllm_patch.py` — Monkey-patches `OpenAIServing` with the custom beam search.
- `beam_star/beam_search.py` — The core beam search loop with adaptive thresholding.
- `beam_star/sampling_params.py` — Extends `BeamSearchParams` with `cum_logprob_threshold`.
- `beam_star/protocol.py` — Converts `ChatCompletionRequest` parameters for beam search.
- `beam_star/stats_utils.py` — `GlobalRunningStats` for online mean/variance/max tracking of logprobs.
- `beam_star/main.py` — Entry point that intercepts `--bs` and applies patches before launching vLLM.
