# Changelog

All notable changes to `nnx-ppo` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-07-01

### Changed
- **Breaking:** `LoggingLevel.TRAINING_ENV_METRICS` is split into
  `LoggingLevel.ENV_METRICS` (the `env/*` subtree from `env_state.metrics`) and
  `LoggingLevel.NETWORK_METRICS` (the `net/*` subtree from a network module's
  `out.metrics`, e.g. entropy, KL, `mu`/`sigma`, forward-model MSE). The old
  flag gated both at once and was misnamed (it included network metrics and
  carried a `TRAINING_` prefix despite now also serving eval). Migrate
  `TRAINING_ENV_METRICS` → `ENV_METRICS | NETWORK_METRICS`. Metric key names are
  unchanged.
- **Breaking:** `LoggingLevel.TRAIN_ROLLOUT_STATS` renamed to
  `LoggingLevel.ROLLOUT_STATS` (the flags serve both training and eval).
- `EvalConfig.logging_level` now defaults to `LoggingLevel.NONE` instead of
  `BASIC`. Eval computes no losses, so the old `LOSSES`/`BASIC` default gated
  nothing and only implied otherwise; eval honours `NETWORK_METRICS` /
  `ENV_METRICS` and always emits the `eval/episode_reward/*` + `eval/lifespan/*`
  headline. Behaviour-identical for the default.
- **Breaking:** all `eval_rollout` metrics are now `eval/`-prefixed
  (`eval/episode_reward/*`, `eval/lifespan/*`, `eval/net/*`, `eval/env/*`), so they
  can never collide with training metrics when merged into one dict per
  iteration. Throughput keys are the exception and stay grouped as
  `throughput/{train,eval,video}_sps`. Update consumers reading
  `episode_reward/mean` etc. to the `eval/`-prefixed names.

### Added
- Activation recording (`nnx_ppo.networks.recording`): an eval/analysis-only
  utility for capturing per-unit activations. `with_recording(net)` returns a
  separate copy of a network (the original is untouched) in which every leaf
  module is wrapped in a `Recorder` that injects the module's forward `output`
  into its `metrics` under the reserved key `ACTIVATION_KEY`; the activation then
  rides the existing `metrics` channel to the top-level call.
  `extract_activations(out.metrics)` pulls them back out, keyed by each module's
  structural path. No per-module code is required, and recording is "off" by
  absence — the unwrapped network is unchanged.
- `PopulationGraph` gains a `record_activations` flag (off by default, enabled by
  `with_recording`): when set it emits every population's post-activation output
  under `ACTIVATION_KEY`. The graph is a special case because its units are
  internal populations (not sub-modules) and it drops its children's metrics.
- `record_activations_rollout(env, networks, n_envs, max_episode_length, key)` in
  `nnx_ppo.algorithms.rollout`: a convenience deterministic rollout that stacks
  per-step activations into `[max_episode_length, n_envs, ...]` arrays (plus the
  pre-step termination mask). Note: `eval_rollout` reduces metrics to scalars and
  is *not* usable for per-unit activations.
- `LoggingLevel.ROLLOUT_OBS` is now functional: it logs the full observation
  pytree (`rollout_batch/obs/*`) during training/distillation. It is a debug aid
  for small-obs envs and is **excluded from `LoggingLevel.ALL`** because it is
  costly for large-obs envs — opt in explicitly.
- Eval now honours `EvalConfig.logging_level`: `eval_rollout` logs `eval/net/*`
  (under `NETWORK_METRICS`) and `eval/env/*` (under `ENV_METRICS`), accumulated
  over the episode, masked by termination and normalised by per-env lifespan.
- `eval/episode_reward/mean` (+`/std`) is an always-on eval headline — the total
  episode return summed across reward keys, averaged over envs — emitted
  regardless of `logging_level`/`logging_percentiles`. It doubles as the
  "did an eval run?" sentinel and gives multi-reward (dict) envs a single
  comparable scalar.

### Fixed
- Eval reward/lifespan keys now use the standard `<name>/<stat>` separator and
  the `eval/` prefix. Previously `episode_reward_mean`/`episode_reward_std` (read
  by several scripts) were **never emitted** — the real keys were
  `episode_reward/mean` (slash) — and `lifespan_mean`/`lifespan_std` were the
  lone underscore-separated outliers. Multi-reward envs now also get a single
  `eval/episode_reward/mean` instead of only per-term subtrees.
- `eval_rollout` no longer crashes for envs whose `reset` returns a non-float
  `done`: the initial state's `done` is cast to float so it matches the dtype the
  scan latches (previously only float-`done` envs worked).

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
