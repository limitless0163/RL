# Reinforcement learning examples

Python 3.11 tutorials from multi-armed bandits through multi-agent reinforcement learning.

## Setup and checks

```bash
python -m pip install -r requirements.txt
python -m ruff check .
python -m ruff format --check .
pyright --pythonpath "$(command -v python)"
python -W error -m compileall -q .
python -W error -m unittest discover -s tests -v
```

Use the same Python environment in VS Code/Pylance and in the terminal. The A100 environment is `/root/anaconda3/envs/limitless/bin/python`.

Each numbered script starts its training experiment when executed. Use `MPLBACKEND=Agg` on a server without a display. Full training runs can take substantial time; the regression tests load definitions without starting those experiments. The warning cleanup was additionally validated by running all 20 numbered examples with reduced training settings and warnings treated as errors. These short runs check execution, not policy convergence or benchmark performance.

## Environment compatibility

Single-agent examples use Gymnasium's `reset()` and five-value `step()` API, current environment IDs, and `reset(seed=...)`. Tensor batches convert NumPy observations to an array before constructing a tensor.

IPPO keeps the Combat environment's ordered list API. The local `envs/combat.py` and drawing helpers are adapted from [boyu-ai/ma-gym](https://github.com/boyu-ai/ma-gym/tree/c24cf4cfb87d542166ddbcd7152e3a404587186a) to use Gymnasium spaces. They no longer depend on the old Gym registration or rendering APIs. The upstream Apache 2.0 license is preserved in `envs/MA_GYM_LICENSE`.

MADDPG uses MPE2's maintained `simple_adversary_v3` parallel environment, with an adapter exposing observations, rewards, and actions in a stable agent order. Its continuous action vectors retain the five movement components expected by the policy. MPE2 replaces the old `multiagent-particle-envs` dependency; learning curves can differ between simulator versions.
