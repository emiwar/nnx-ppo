# Changelog

All notable changes to `nnx-ppo` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-06-03

Initial PyPI release.

### Added
- Stateful-network PPO training loop (`nnx_ppo.algorithms.ppo.train_ppo`).
- Network containers (`Sequential`, `Parallel`, `Concat`, `Splitter`) and the
  two-port `PPOAdapter`.
- Built-in layers: `Dense`, `LSTM`, `AR1VariationalBottleneck`, `Normalizer`,
  `Delay`, sampling layers, and graph-population utilities.
- Rollout machinery with per-environment state reset and `update_statistics`
  hook for stats-bearing modules.
- Orbax-based checkpointing.
- Distillation utility (`nnx_ppo.algorithms.distillation`).
- Documentation site at <https://nnx-ppo.readthedocs.io>.
