# Changelog

All notable changes to `nnx-ppo` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] — 2026-06-09

### Added
- `LoggingLevel.THROUGHPUT` — emits `throughput/train_sps`,
  `throughput/eval_sps`, and `throughput/video_sps` (env + render),
  with `jax.block_until_ready` barriers so the numbers reflect
  device-side wall-clock rather than JAX dispatch latency. Included in
  `LoggingLevel.ALL`.
- `losses/clipping_fraction` under `LoggingLevel.ACTOR_EXTRA` — the
  fraction of samples whose likelihood ratio left the PPO clip range
  during the gradient phase. Tree-mapped, so it works with multi-actor
  / multi-agent loglikelihoods.

### Changed
- `RewardScalingWrapper` no longer depends on `mujoco_playground`; it
  is now typed against the local `RLEnv` / `EnvState` protocols in
  `nnx_ppo.algorithms.types`.
- The `[playground]` extra has been removed. `playground` and
  `warp-lang` are now part of the `[dev]` and `[examples]` extras.
- Minimum `flax` is now `0.12.7`.
- License metadata switched to PEP 639 SPDX form
  (`license = "BSD-3-Clause"` + `license-files = ["LICENSE"]`);
  requires `setuptools>=77.0` at build time.

### Fixed
- `PopulationGraph` no longer exposes its build-time registries as a
  second set of `nnx.Param`s — newer Flax versions reflected through
  the underscore-prefixed dicts, which tripped `nnx.jit`'s
  consistent-aliasing check.

### Removed
- `correlations/action_ll` (under `ACTOR_EXTRA`) — was only emitted
  for 1-D action spaces and never fired in multi-actuator setups.

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
