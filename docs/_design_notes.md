# Design notes for the redesigned `networks` module

Reference notes for writing the Sphinx documentation. Not user-facing; the actual
narrative guide / API pages will be in `docs/` proper.

Plan file: `/home/emil/.claude/plans/i-want-to-redesign-tender-rabbit.md`

---

## Core abstractions

### `StatefulModule` ([networks/types.py](../nnx_ppo/networks/types.py))

Abstract base for every composable layer. Two kinds of state:

- **NNX module state** (params, RNG streams) — managed by NNX; not reset on env reset.
- **Explicit carry state** — passed in/out of `__call__`; reset when env resets via `reset_state`.

Signature (KEY — this is the new shape):
```python
def __call__(self, module_state, x, *, context: Context = Context.INFERENCE) -> StatefulModuleOutput: ...
def initialize_state(self, batch_size) -> ModuleState: ...
def reset_state(self, prev_state) -> ModuleState: ...
```

`StatefulModuleOutput` fields: `next_state`, `output`, `regularization_loss`, `metrics`.

### `PPONetwork`

Abstract base for top-level PPO networks. Same signature except `__call__` also takes an
`Optional[raw_action]` and returns `PPONetworkOutput`. Used to be `PPOActorCritic`; now
prefer `PPOAdapter`.

### `PPONetworkOutput`

Fields: `actions`, `raw_actions`, `loglikelihoods`, `regularization_loss`, `value_estimates`,
`metrics`, **`distribution_params`** (new). The last field exposes pre-sampling distribution
parameters (mu/sigma for `NormalTanhSampler`) keyed by sampler name — populated by
`PPOAdapter` so distillation can read student/teacher distributions without an extra forward.

---

## The `Context` enum — central organizing concept

```python
class Context(enum.Enum):
    ROLLOUT      = "rollout"        # unroll_env: drive env, collect data
    LOSS_REPLAY  = "loss_replay"    # ppo_loss: re-run with stored raw_action for gradients
    INFERENCE    = "inference"      # eval_rollout / teacher / ad-hoc; default
    STATS_UPDATE = "stats_update"   # post-loss: re-run rollout to fold inputs into running stats
```

**Threading rule.** `context` is keyword-only, threaded by every container (`Sequential`,
`Parallel`, `Concat`, `Splitter`, `Flattener`, `Delay`, `PPOActorCritic`, `PPOAdapter`) to its
children. Leaf modules that vary by context read the kwarg directly.

**Default `INFERENCE` is the safe fallback.** Forgetting to pass `context=` somewhere
produces no surprises — no stats updates, deterministic-ish behaviour.

### Per-module behaviour table

| Module | ROLLOUT | LOSS_REPLAY | INFERENCE | STATS_UPDATE |
|---|---|---|---|---|
| `Normalizer` | normalize with live params; no update | normalize; no update | normalize; no update | normalize with live params; **inline Welford** updates `mean`/`M2`/`counter` |
| `ActionSampler` | sample (or mean if `deterministic`) | use stored `raw_action`; RNG advances | per-instance `deterministic` decides | use stored `raw_action` (same as LOSS_REPLAY) |
| `VariationalBottleneck` | sample fresh | sample with carried RNG | sample fresh | sample with carried RNG (reproduces rollout activations) |
| `Delay`, `Dense`, `LSTM`, containers | no context-dependent behaviour | — | — | — |

### Rule for writes inside `__call__`

CLAUDE.md was updated:

> **No writes to NNX variables that affect forward output, unless `context == Context.STATS_UPDATE`.**

`LOSS_REPLAY` / `ROLLOUT` / `INFERENCE` are all pure forward passes. Only `STATS_UPDATE` may
mutate state-affecting variables. This is what makes rollout/replay consistent (same params
across rollout and loss within a training step; stats only change between training steps).

---

## Composition layer

### `Sequential([...])` — chain of layers
State is a list matching layer order. Threads `context` to each layer in turn.

### `Concat(**components)` — pytree input → concat outputs along last axis
Input must be a dict matching component keys. Each component sees its corresponding sub-input.
Outputs concatenated along last axis. State is `dict[str, ModuleState]`.

### `Parallel(**components)` — same input, dict of outputs *(new)*
Same input fed to every named component. Output is a dict keyed by component name.
Used to fan out: `Parallel(action_params=actor, value=critic)` gives `{"action_params": ..., "value": ...}`.

### `Splitter(**sizes)` — split a flat tensor into named slices along the last axis *(new)*
`Splitter(action=2*A, value=1)` takes `(B, 2A+1)` and emits `{"action": (B, 2A), "value": (B, 1)}`.
With a single keyword (`Splitter(action_params=N)`) it just relabels the input as a dict.

### `Flattener()` — pytree leaves → concat along last axis

---

## Leaf modules

### `Dense(in_features, out_features, rngs, activation=None)`
Thin wrapper around `nnx.Linear`. Stateless carry (`()`).

### `LSTM(in_features, hidden_features, rngs, ...)`
Wraps `nnx.OptimizedLSTMCell` / `nnx.LSTMCell`. State is `(h, c)` tuple. `reset_state` zeros
(or restores trainable initial state).

### `Delay(sample_input, k_steps, initial_value=0.0)` *(new — promoted from vnl-experiments)*
k-step delay layer. Output at time t = input at time t-k. Carry state:
`{"buffer": <pytree, leaves [B, k_steps, *leaf]>, "idx": <[B] int32>}`. Reset zeros both.

Composable anywhere — Sequential layer, connection transform in PopulationGraph, etc.
Replaces the old `DelayedObsNetwork` wrapper. The wrapper is now equivalent to
`Sequential([Delay(obs_shape, k), inner_network])`.

### `Normalizer(shape)` — running mean/std normalizer, context-aware
Forward standardises with live `mean`/`M2`/`counter`. Updates inline (per-step batched Welford)
only when `context == STATS_UPDATE`. Frozen in all other contexts.

Legacy `update_statistics(rollout, total_steps)` method retained as a `@deprecated` shim for
out-of-tree callers that haven't migrated; folds whole-rollout into stats in one shot via
`_welford_full_rollout`. To be removed after experiment migration.

**Placeable anywhere** — behind a `Delay`, inside a `Sequential`, inside a graph node — and
sees the same activations during the STATS_UPDATE replay as it did during rollout.

### `VariationalBottleneck(latent_size, rng, kl_weight, min_std)`
Reparameterised normal sample, KL against standard normal. State carries per-batch RNG keys.

### `AR1VariationalBottleneck(latent_size, rng, kl_weight, min_std, ar1_weight, backprop_through_time)`
Adds first-order AR1 smoothness penalty on top of the variational KL. State carries
RNG keys and `last_z`. Uses NaN sentinel in `last_z` to detect reset boundaries.

---

## PPO machinery — now in `algorithms/`

Sampler files moved out of `networks/` to make the `networks/` package general-purpose.

### `ActionSampler` (abstract) and `NormalTanhSampler` ([algorithms/distributions.py](../nnx_ppo/algorithms/distributions.py))

Moved from `networks/sampling_layers.py`. The old path is a back-compat re-export shim with a
`DeprecationWarning` on import.

`NormalTanhSampler`: normal distribution followed by tanh. Sample stochastically (or use mean
if instance `deterministic=True`). Computes log-likelihood with tanh-jacobian correction.
Carries entropy bonus through `regularization_loss`. Exposes `{"mu", "sigma"}` in metrics —
these become `PPONetworkOutput.distribution_params[name]` via `PPOAdapter`.

`deterministic` is now a *per-instance* flag with an honest meaning (use mean instead of
sampling). It's no longer overloaded with stats-update semantics.

### `PPOAdapter(inner, action_specs, value_specs)` ([algorithms/adapter.py](../nnx_ppo/algorithms/adapter.py))

The replacement for `PPOActorCritic`. Wraps any `StatefulModule` whose output is a **dict of
named heads**.

```python
PPOAdapter(
    inner=trunk,                              # StatefulModule emitting a dict
    action_specs={"action_params": sampler},  # name -> ActionSampler
    value_specs="value",                      # str or list[str] or dict
)
```

Behaviour:
- Runs `inner(state, obs, context=context)`, gets a dict of head outputs.
- For each entry in `action_specs`: runs the sampler on `inner_output[name]`. If `raw_action`
  was passed (as a dict, or a bare array in single-head mode), threads it to the sampler.
- For each name in `value_specs`: reads `inner_output[name]`, squeezes a trailing length-1
  axis if present.
- Sums `regularization_loss` from inner + every sampler.
- Populates `PPONetworkOutput.distribution_params[name] = sampler.metrics` (for distillation).

**Single-head convenience.** If `action_specs` has one entry, `actions`/`raw_actions`/
`loglikelihoods` are unwrapped from the single-key dict into bare arrays — matching the
shape `PPOActorCritic` used to produce. Same for `value_specs` if it's a string or has a
single entry. Multi-head mode keeps them as dicts.

State: `{"inner": <inner state>, "samplers": {name: <sampler state>}}`.

### `PPOActorCritic` — convenience subclass of `PPOAdapter`

`PPOActorCritic(actor, critic, action_sampler, preprocessor=None)` is the standard
one-actor / one-critic convenience. After the refactor it is a **thin subclass of
`PPOAdapter`** (~15 LoC in [containers.py](../nnx_ppo/networks/containers.py)) that just
constructs the inner module:

```python
inner = Parallel(action_params=actor, value=critic)
if preprocessor is not None:
    inner = Sequential([preprocessor, inner])
super().__init__(
    inner=inner,
    action_specs={"action_params": action_sampler},
    value_specs="value",
)
```

The subclass also exposes `self.actor`, `self.critic`, `self.action_sampler`, and
`self.preprocessor` as direct attributes for backwards-compatible introspection
(parameter logging in [algorithms/metrics.py](../nnx_ppo/algorithms/metrics.py),
checkpointing tests, factories tests). NNX handles the shared references — the same
module instance lives both at `self.actor` and inside `self.inner.components["action_params"]`.

**State shape changed.** Old `PPOActorCritic` had
`{"actor": ..., "critic": ..., "action_sampler": ..., "preprocessor": ...}`; the new
subclass inherits `PPOAdapter`'s `{"inner": ..., "samplers": {"action_params": ...}}`.
Existing call sites that pass state opaquely (rollout, loss, eval) are unaffected.
Sites that index by key (checkpointing tests, factories tests) were updated to the
new path.

This restores the two-tier mental model:
- **Tutorial / basic usage**: `PPOActorCritic(actor, critic, sampler, preprocessor=Normalizer(...))`.
- **Modular / multi-head usage**: drop down to bare `PPOAdapter`.

---

## Training loop integration ([algorithms/rollout.py](../nnx_ppo/algorithms/rollout.py), [algorithms/ppo.py](../nnx_ppo/algorithms/ppo.py))

### `unroll_env(..., *, context=Context.ROLLOUT)`
Default `ROLLOUT`. Threads context to `single_transition` which threads to `networks(...)`.

### `ppo_loss(...)` 
Internal `step_network` function now passes `context=Context.LOSS_REPLAY` to the network on
every scan step and the last-obs call. This locks in the rollout-vs-replay consistency
property.

### `eval_rollout(...)` and `eval_rollout_for_render_scan(...)`
Both pass `context=Context.INFERENCE`.

### `ppo_step` 
**Not yet migrated to the STATS_UPDATE replay pass.** Currently still calls the legacy
`training_state.networks.update_statistics(rollout_data, total_steps)`. The legacy path
works fine for top-level Normalizers via the deprecated `update_statistics(rollout, ...)`
shim. Plan: replace this with a `set_context(STATS_UPDATE)` + `_replay_rollout_for_stats`
call once the vnl-experiments NerveNet variants are rewritten on the new context-based API
(otherwise their unmigrated `__call__` signatures would break).

---

## Migration story

### What stays the same
- `StatefulModule.__call__`'s positional args (`state`, `x`). Plus `raw_action` for `PPONetwork`.
- All the existing leaf modules' construction APIs.
- `PPOActorCritic` still works.
- The legacy `network.update_statistics(rollout, total_steps)` post-rollout hook still works
  (but is deprecated).

### What changes
- Every `__call__` gains a keyword-only `context` parameter (defaults `INFERENCE` — safe).
- `Normalizer` now updates inline during `STATS_UPDATE` context; the post-hoc
  `update_statistics(rollout, ...)` call still works as a deprecated shim.
- `NormalTanhSampler` / `ActionSampler` import path changes from
  `nnx_ppo.networks.sampling_layers` → `nnx_ppo.algorithms.distributions`. Old path
  still works but warns.
- New `Parallel`, `Splitter`, `Delay` modules in `networks/`.
- New `PPOAdapter` in `algorithms/`, designed to replace `PPOActorCritic`.

### Deprecation surface
PEP 702 `@deprecated` decorator (Python 3.13+) applied to:
- `StatefulModule.update_statistics` (base method)
- `Normalizer.update_statistics` (legacy shim)
- `Normalizer._welford_full_rollout` (helper)

IDEs flag call sites with hints / strikethroughs. Runtime emits `DeprecationWarning`.
Import-time `DeprecationWarning` on `nnx_ppo.networks.sampling_layers`.

---

## Still to do (in priority order)

1. **`PopulationGraph` subpackage** ([networks/graph/](../nnx_ppo/networks/graph/) — new):
   - `Population` (internal `nnx.Module`): name, input_size, output_size, compute, activation, max_outgoing_delay.
   - `Connection` (internal `nnx.Module`): src, dst, transform (default `Dense`, linear), delay.
   - `PopulationGraph(StatefulModule)`: `add_population(name, output_size, ..., input_from=None)`,
     `connect(src, dst, transform=None, delay=0)`, `add_output(name, source, head=None)`,
     `finalize()` (topo sort + shape check + cycle detection — delay-0 cycles are hard errors —
     plus per-population `max_outgoing_delay` computation for shared delay buffers).
   - Sum-integration only for v1. Activation lives on the population, not the connection.
   - Per-population shared delay buffer of length `max(1, max_outgoing_delay)`. Stores
     post-activation outputs. Delayed connections read from this buffer at their offset.
   - Input populations declare `input_from="obs_key"` to read from `obs[key]` instead of
     accumulating incoming connections.

2. **Rewrite `enc_dec.py` and `nervenet_style_v3.py`** in vnl-experiments on the new API
   (uses `PPOAdapter` + `Parallel`/`Splitter` for enc_dec, `PopulationGraph` for nervenet).
   Drop the legacy `update_statistics` shim once these have migrated.

3. **`_replay_rollout_for_stats` helper + ppo_step migration** — adds the `STATS_UPDATE`
   replay pass after the gradient phase, removes the legacy `networks.update_statistics(rollout)`
   call. Cleanest to land alongside the experiment rewrites.

4. **Sphinx docs**: see "Documentation pages to write" below.

---

## Documentation pages to write

(These should land in `docs/` proper; this notes file is just scaffolding.)

### Update [docs/api/networks.rst](api/networks.rst)
Cover the new public surface: `StatefulModule`, `Context`, `Sequential`, `Concat`,
`Splitter`, `Parallel`, `Flattener`, `Delay`, `Normalizer`, `Dense`, `LSTM`,
`VariationalBottleneck`, `AR1VariationalBottleneck`, `PopulationGraph` (once it exists).
Drop entries for `PPOActorCritic` and `sampling_layers`.

### Update [docs/api/algorithms.rst](api/algorithms.rst)
Add: `PPOAdapter` (the new central composition primitive), `ActionSampler` /
`NormalTanhSampler` (moved here).

### New narrative page: `docs/networks_guide.rst` (suggested filename)

Topics to cover, with worked examples:

1. **The two composition styles** — `Sequential`/`Concat`/`Parallel`/`Splitter` for stacks,
   `PopulationGraph` for graphs with multiple connections and recurrent loops. Show when to
   use each.

2. **The `Context` enum and threading rule.** Worked example: building a network and stepping
   through ROLLOUT → LOSS_REPLAY → STATS_UPDATE phases, showing what each does.

3. **Worked example: encoder-decoder (modeled on enc_dec.py)** with `PPOAdapter`:
   ```python
   trunk = Sequential([
       Normalizer(obs_size),
       Parallel(
           action_params=Sequential([encoder_actor_stack, decoder_actor_stack, action_head]),
           value=Sequential([Flattener(), value_mlp]),
       ),
   ])
   net = PPOAdapter(
       inner=trunk,
       action_specs={"action_params": NormalTanhSampler(...)},
       value_specs="value",
   )
   ```

4. **Worked example: NerveNet-style modular network** with `PopulationGraph`:
   declarative `add_population` + `connect`. Discuss the semantic change (one nonlinearity
   per population step rather than two — old checkpoints don't transfer).

5. **`Delay` placement**. Show input delay (`Sequential([Delay(obs_shape, k), inner])`),
   per-connection delay (`graph.connect(a, b, delay=N)`), and recurrent self-loops
   (`graph.connect(a, a, delay=1)`).

6. **`Normalizer` placement**. Show top-level (typical), behind a `Delay`, embedded inside a
   graph population. Explain the STATS_UPDATE replay model: the post-loss pass re-runs the
   rollout in `STATS_UPDATE` context with stored `raw_action`s; each `Normalizer` sees the
   same activations it saw during rollout because all preceding modules are deterministic
   under stored RNG. No explicit walker; correctness by construction.

7. **`PPOAdapter` details**: action_specs / value_specs shapes, multi-head dict mode vs.
   single-head unwrapped mode, the `distribution_params` field on output and its use in
   distillation.

8. **Distillation pattern**: student in `ROLLOUT`/`LOSS_REPLAY`, teacher in `INFERENCE`
   with `deterministic=True` sampler. Read `distribution_params` for KL/MSE losses.

9. **Migration guide**. For each old pattern, show the new equivalent. Specifically:
   - `PPOActorCritic(actor, critic, sampler, preprocessor=Normalizer(...))` →
     `PPOAdapter(inner=Sequential([Normalizer(...), Parallel(action_params=actor, value=critic)]), ...)`.
   - `DelayedObsNetwork(net, k, sample_obs)` → `Sequential([Delay(sample_obs, k), net])`.
   - `from nnx_ppo.networks.sampling_layers import NormalTanhSampler` →
     `from nnx_ppo.algorithms.distributions import NormalTanhSampler`.

10. **The "no Variable writes in `__call__` unless `STATS_UPDATE`" rule**. Why it exists
    (rollout/replay consistency). What modules are allowed to write (currently just
    `Normalizer`).

### Wire new pages into [docs/index.rst](index.rst) `toctree`.

---

## Design decisions worth recording

These are settled (after substantial back-and-forth with the user — see plan file for the
full discussion). Worth surfacing in docs so future contributors don't relitigate.

1. **Context as a kwarg, not an attribute** (user chose this over a recursive `set_context`
   setter). Explicit at every call site beats hidden module state. Cost: every container
   threads it, but that's bounded and one-line.

2. **Four contexts, not three.** `STATS_UPDATE` is structurally identical to `LOSS_REPLAY`
   but distinguishable so `Normalizer` knows when to update. Considered collapsing into
   "training/eval" but rejected: those terms mean too many things.

3. **Per-instance `deterministic` flag stays orthogonal to context.** It's the legitimate
   "in eval, use mean" knob. Not overloaded with stats semantics.

4. **Stats updated post-loss, not during rollout.** Considered inline-during-rollout (saves
   one forward pass) and pending-accumulator (avoids one forward pass via dual variables),
   both rejected: inline-during-rollout breaks rollout/replay consistency; pending
   accumulators add real conceptual complexity. The extra forward pass is small compared to
   env stepping and `n_epochs × n_minibatches` loss replays.

5. **`Activation` is on the destination population, not the connection** (in
   `PopulationGraph`). Cleaner semantics — one nonlinearity per population step rather than
   two — and matches the neuroscience meaning ("transfer function of the unit").

6. **`PopulationGraph` is a peer of `Sequential`, not a replacement.** Use Sequential for
   feedforward stacks (cleaner there); use PopulationGraph when you actually need graph
   topology, recurrent connections, or per-connection delays.

7. **Samplers and `PPOAdapter` belong in `algorithms/`, not `networks/`.** They're
   policy-distribution machinery, not general-purpose network layers. The `networks/`
   package could be released as a standalone library; `algorithms/` is PPO-specific.

---

## File map (as of this checkpoint)

```
nnx_ppo/
├── networks/
│   ├── types.py            # StatefulModule, PPONetwork, Context, PPONetworkOutput
│   ├── containers.py       # Sequential, Concat, Parallel, Splitter, Flattener, PPOActorCritic (legacy)
│   ├── feedforward.py      # Dense
│   ├── recurrent.py        # LSTM
│   ├── delay.py            # Delay (new)
│   ├── normalizer.py       # Normalizer (context-aware, with legacy shim)
│   ├── variational.py      # VariationalBottleneck, AR1VariationalBottleneck
│   ├── factories.py        # make_mlp, make_mlp_actor_critic
│   ├── sampling_layers.py  # back-compat re-export shim (deprecated)
│   └── graph/              # PopulationGraph (TODO)
└── algorithms/
    ├── types.py            # Transition, TrainingState, etc.
    ├── rollout.py          # unroll_env (context=ROLLOUT default), eval_rollout (INFERENCE)
    ├── ppo.py              # ppo_step, ppo_loss (LOSS_REPLAY threaded into scan)
    ├── distributions.py    # ActionSampler, NormalTanhSampler (new home)
    ├── adapter.py          # PPOAdapter (new)
    ├── checkpointing.py    # (unchanged)
    └── config.py           # (unchanged)
```

Test status at checkpoint: 79 tests pass (72 networks + 7 adapter + 44 algorithms = 123 total
including the existing algorithms tests, all green). Zero regressions in any pre-existing
test.
