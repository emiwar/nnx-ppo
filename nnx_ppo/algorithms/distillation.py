"""Policy distillation training algorithm.

Trains a student StatefulModule (whose forward output is a
StatefulModuleOutput) to imitate a frozen teacher.

The algorithm (Policy Distillation, Rusu et al. 2015):
  1. Roll out the environment using student actions.
  2. During rollout, also run the teacher to collect its rollout_extras
     (which, with the teacher in eval mode, contains its action mean at
     the sampler positions).
  3. Train the student to maximise log p_student(mu_teacher | obs), i.e.
     minimise the NLL of the teacher's action mean under the student's
     distribution. Equivalent to minimising KL(teacher || student) up to
     the constant H(teacher).
  4. Repeat for n_epochs * n_minibatches gradient updates per rollout.

The teacher is run in eval (deterministic) mode so that the sampler's
emitted raw_action equals the mean. Teacher parameters are never
updated; the teacher is not passed to the loss function — only its
pre-computed rollout_extras (stored in DistillationTransition) is used
there as the student's loss-replay channel.

Constraint: teacher and student must have isomorphic state / rollout_extras
trees (i.e. the same architectural skeleton with samplers at matching
positions), because the teacher's rollout_extras is fed directly into the
student's __call__ during loss replay.
"""

from typing import Any, Optional
from collections.abc import Callable
import dataclasses
import functools

import numpy as np

from flax import nnx
import jax
import jax.numpy as jp
from jaxtyping import Array, Float, Integer, PRNGKeyArray
import optax

from nnx_ppo.networks.types import ModuleState, StatefulModule
from nnx_ppo.algorithms import rollout
from nnx_ppo.algorithms.rollout import tree_where
from nnx_ppo.algorithms.types import (
    DistillationTransition,
    DistillationState,
    LoggingLevel,
    RLEnv,
)
from nnx_ppo.algorithms.config import (
    DistillationTrainConfig,
    DistillationConfig,
    EvalConfig,
    VideoConfig,
    VideoData,
    DistillationTrainResult,
)
from nnx_ppo.algorithms.metrics import _log_metric
from nnx_ppo.algorithms.ppo import _should_run


def default_distillation_config() -> DistillationTrainConfig:
    return DistillationTrainConfig()


def distillation_single_transition(
    env: RLEnv,
    teacher: StatefulModule,
    student: StatefulModule,
    carry: tuple[ModuleState, ModuleState, Any],
    rng_keys_for_env_reset: Any,
) -> tuple[tuple[ModuleState, ModuleState, Any], DistillationTransition]:
    """Single environment step for distillation rollout.

    Runs both teacher and student on the current observation.
    The student's actions drive the environment forward.
    The teacher's output is stored as the distillation target.
    """
    student_state, teacher_state, env_state = carry

    student_out = student(student_state, env_state.obs)
    teacher_out = teacher(teacher_state, env_state.obs)
    next_student_state = student_out.next_state
    next_teacher_state = teacher_out.next_state
    student_output = student_out.output

    next_env_state = jax.vmap(env.step)(env_state, student_output.actions)

    transition = DistillationTransition(
        obs=env_state.obs,
        student_output=student_output,
        rewards=next_env_state.reward,
        done=next_env_state.done.astype(bool),
        truncated=next_env_state.info.get(
            "truncated", jp.zeros(next_env_state.done.shape, bool)
        ).astype(bool),
        next_obs=next_env_state.obs,
        metrics={
            "env": next_env_state.metrics,
            "student": student_out.metrics,
        },
        student_rollout_extras=student_out.rollout_extras,
        teacher_rollout_extras=teacher_out.rollout_extras,
    )

    done = transition.done
    reset_env_states = jax.vmap(env.reset)(rng_keys_for_env_reset)
    next_env_state = tree_where(done, reset_env_states, next_env_state)

    reset_student_states = student.reset_state(next_student_state)
    next_student_state = tree_where(done, reset_student_states, next_student_state)

    reset_teacher_states = teacher.reset_state(next_teacher_state)
    next_teacher_state = tree_where(done, reset_teacher_states, next_teacher_state)

    return (next_student_state, next_teacher_state, next_env_state), transition


def distillation_unroll_env(
    env: RLEnv,
    env_state: Any,
    teacher: StatefulModule,
    student: StatefulModule,
    student_state: ModuleState,
    teacher_state: ModuleState,
    unroll_length: int,
    rng_key_for_env_reset: PRNGKeyArray,
) -> tuple[ModuleState, ModuleState, Any, DistillationTransition]:
    """Roll out the environment for distillation training.

    Runs both teacher and student at every step. The student drives the env;
    the teacher produces the target distributions stored in the returned rollout.

    Returns:
        (final_student_state, final_teacher_state, final_env_state, rollout_data)
    """
    batch_size = env_state.done.shape[0]
    rng_keys_for_env_reset = jax.random.split(
        rng_key_for_env_reset, (unroll_length, batch_size)
    )

    step = functools.partial(distillation_single_transition, env)

    (final_student_state, final_teacher_state, final_env_state), rollout_data = nnx.scan(
        step,
        in_axes=(
            nnx.StateAxes({...: nnx.Carry}),  # teacher NNX module
            nnx.StateAxes({...: nnx.Carry}),  # student NNX module
            nnx.Carry,                          # (student_state, teacher_state, env_state)
            0,                                  # rng_keys_for_env_reset, shape [T, B]
        ),
        out_axes=(nnx.Carry, 0),
        length=unroll_length,
    )(teacher, student, (student_state, teacher_state, env_state), rng_keys_for_env_reset)

    return final_student_state, final_teacher_state, final_env_state, rollout_data


def distillation_loss(
    student: StatefulModule,
    student_state: ModuleState,
    rollout_data: DistillationTransition,
    logging_level: LoggingLevel,
) -> tuple[Float[Array, ""], dict[str, Any]]:
    """Compute distillation loss.

    Replays the student over the rollout trajectory and computes the negative
    log-likelihood (NLL) of the teacher's action mean under the student's
    distribution. This is equivalent to minimising KL(teacher || student) up
    to the constant H(teacher).

    The teacher was run in eval (deterministic) mode during rollout, so its
    emitted rollout_extras carries the teacher mean at every sampler position.
    Passing it as the student's `rollout_extras` makes the student's sampler
    compute `log p_student(mu_teacher | obs)`.

    Args:
        student: Student network (gradient target).
        student_state: Student carry state at the start of the minibatch rollout.
        rollout_data: Pre-computed rollout including teacher rollout_extras.
                      Treated as stop-gradient constants.
        logging_level: Controls which metrics are returned.

    Returns:
        (total_loss, loss_metrics)
    """
    rollout_data = jax.lax.stop_gradient(rollout_data)

    @jax.vmap
    def reset_net_state(done, state):
        return jax.lax.cond(done, student.reset_state, lambda x: x, state)

    def step_network(student, net_state, obs, done, teacher_rollout_extras):
        out = student(net_state, obs, teacher_rollout_extras)
        net_state = reset_net_state(done, out.next_state)
        return net_state, out

    time_scan = nnx.scan(
        step_network,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry, 0, 0, 0),
        out_axes=(nnx.Carry, 0),
    )

    _, student_outs = time_scan(
        student,
        student_state,
        rollout_data.obs,
        rollout_data.done,
        rollout_data.teacher_rollout_extras,
    )

    # student_outs.output.loglikelihoods = log p_student(mu_teacher | obs).
    # Either a single array (single-head) or a dict (multi-head); use
    # tree ops to handle both.
    per_head_nll = jax.tree.map(
        lambda ll: -jp.mean(ll), student_outs.output.loglikelihoods
    )
    nll_loss = jax.tree.reduce(jp.add, per_head_nll)

    # Student regularization (entropy bonus, AR1 smoothness, etc.) is preserved.
    # Teacher's regularization_loss is intentionally ignored.
    regularization_losses = jax.tree.map(jp.mean, student_outs.regularization_loss)
    regularization_loss = jax.tree.reduce(jp.add, regularization_losses)

    total_loss = nll_loss + regularization_loss

    loss_metrics: dict[str, Any] = {}
    if LoggingLevel.LOSSES in logging_level:
        loss_metrics["losses/distillation_nll"] = nll_loss
        loss_metrics["losses/regularization"] = regularization_loss

    return total_loss, loss_metrics


def distillation_step(
    env: RLEnv,
    teacher: StatefulModule,
    distillation_state: DistillationState,
    n_envs: int,
    rollout_length: int,
    n_epochs: int,
    n_minibatches: int,
    logging_level: LoggingLevel = LoggingLevel.LOSSES,
    logging_percentiles: Optional[tuple[int, ...]] = None,
) -> tuple[DistillationState, dict[str, Any]]:
    """Single distillation training step: rollout + multi-epoch gradient updates.

    Args:
        env: Training environment.
        teacher: Frozen teacher network (not modified, passed as external dep).
        distillation_state: Current training state.
        n_envs: Number of parallel environments.
        rollout_length: Steps per rollout.
        n_epochs: Gradient update epochs per rollout.
        n_minibatches: Number of minibatches per epoch.
        logging_level: Controls which metrics to log.
        logging_percentiles: Percentile levels for metric aggregation.

    Returns:
        (updated_distillation_state, metrics)
    """
    reset_key, new_key = jax.random.split(distillation_state.rng_key)

    # Phase 1: Rollout with both teacher and student
    next_student_state, next_teacher_state, next_env_state, rollout_data = (
        distillation_unroll_env(
            env,
            distillation_state.env_states,
            teacher,
            distillation_state.student,
            distillation_state.student_states,
            distillation_state.teacher_states,
            rollout_length,
            reset_key,
        )
    )

    # Phase 2: Multiple gradient updates over minibatches
    grad_fn = nnx.grad(distillation_loss, has_aux=True)

    total_iterations = n_epochs * n_minibatches
    minibatch_size = n_envs // n_minibatches

    def get_epoch_indices(epoch_idx):
        shuffle_key = jax.random.fold_in(new_key, epoch_idx)
        perm = jax.random.permutation(shuffle_key, n_envs)
        return perm.reshape(n_minibatches, minibatch_size)

    # Shape: (n_epochs * n_minibatches, minibatch_size)
    all_indices = jax.vmap(get_epoch_indices)(jp.arange(n_epochs))
    all_indices = all_indices.reshape(total_iterations, minibatch_size)

    def update_step(
        student: StatefulModule,
        optimizer: nnx.Optimizer,
        inds: Integer[Array, "minibatch"],
    ):
        minibatch_data = jax.tree.map(lambda x: x[:, inds], rollout_data)
        student_state_subset = jax.tree.map(
            lambda x: x[inds], distillation_state.student_states
        )
        grads, loss_metrics = grad_fn(
            student=student,
            student_state=student_state_subset,
            rollout_data=minibatch_data,
            logging_level=logging_level,
        )
        optimizer.update(student, grads)
        return loss_metrics

    scan_update = nnx.scan(
        update_step,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.StateAxes({...: nnx.Carry}), 0),
        out_axes=0,
        length=total_iterations,
    )
    loss_metrics = scan_update(
        distillation_state.student, distillation_state.optimizer, all_indices
    )

    total_steps = distillation_state.steps_taken + rollout_length * n_envs

    # Phase 3: fold the student's rollout_extras into any stats-bearing
    # modules (Normalizer, ...). Mirrors ppo.py's post-update call.
    distillation_state.student.update_statistics(rollout_data.student_rollout_extras)

    # Build metrics dict
    metrics: dict[str, Any] = {}
    for k, v in loss_metrics.items():
        _log_metric(metrics, k, v, logging_percentiles)

    if LoggingLevel.TRAIN_ROLLOUT_STATS in logging_level:
        _log_metric(
            metrics, "rollout_batch/reward", rollout_data.rewards, logging_percentiles
        )
        _log_metric(
            metrics,
            "rollout_batch/action",
            rollout_data.student_output.actions,
            logging_percentiles,
        )
        metrics["rollout_batch/done_rate"] = rollout_data.done.mean()
        metrics["rollout_batch/truncation_rate"] = rollout_data.truncated.mean()

    if LoggingLevel.TRAINING_ENV_METRICS in logging_level:
        for k, v in rollout_data.metrics.items():
            _log_metric(metrics, k, v, logging_percentiles)

    metrics["total_steps"] = total_steps

    distillation_state = distillation_state.replace(
        student_states=next_student_state,
        teacher_states=next_teacher_state,
        env_states=next_env_state,
        rng_key=new_key,
        steps_taken=total_steps,
    )

    return distillation_state, metrics


def new_distillation_state(
    env: RLEnv,
    teacher: StatefulModule,
    student: StatefulModule,
    n_envs: int,
    seed: int,
    learning_rate: float = 1e-4,
    gradient_clipping: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> DistillationState:
    """Create initial DistillationState.

    Args:
        env: Training environment.
        teacher: Frozen teacher network (state is initialised but not trained).
        student: Student network to be trained.
        n_envs: Number of parallel environments.
        seed: Random seed.
        learning_rate: Learning rate for the student optimizer.
        gradient_clipping: If set, clip gradients by global norm.
        weight_decay: If set, use AdamW with this weight decay.

    Returns:
        Fresh DistillationState with zero steps_taken.
    """
    key = jax.random.key(seed)
    key, training_key = jax.random.split(key)

    env_init_keys = jax.random.split(key, n_envs)
    env_states = nnx.vmap(env.reset)(env_init_keys)

    student_states = student.initialize_state(n_envs)
    teacher_states = teacher.initialize_state(n_envs)

    # Optimizer tracks only the student's nnx.Param variables.
    optimizer_chain_links = []
    if gradient_clipping is not None:
        optimizer_chain_links.append(optax.clip_by_global_norm(gradient_clipping))
    if weight_decay is None:
        optimizer_chain_links.append(optax.adam(learning_rate=learning_rate))
    elif isinstance(weight_decay, bool) and weight_decay:
        optimizer_chain_links.append(optax.adamw(learning_rate=learning_rate))
    else:
        optimizer_chain_links.append(
            optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        )
    optimizer = nnx.Optimizer(student, optax.chain(*optimizer_chain_links), wrt=nnx.Param)

    return DistillationState(
        student=student,
        student_states=student_states,
        teacher_states=teacher_states,
        env_states=env_states,
        optimizer=optimizer,
        rng_key=training_key,
        steps_taken=jp.array(0.0),
    )


def train_distillation(
    env: RLEnv,
    teacher: StatefulModule,
    student: StatefulModule,
    config: Optional[DistillationTrainConfig] = None,
    *,
    total_steps: Optional[int] = None,
    seed: Optional[int] = None,
    log_fn: Optional[Callable[[dict[str, Any], int], None]] = None,
    video_fn: Optional[Callable[[VideoData], None]] = None,
    checkpoint_fn: Optional[Callable[[DistillationState, int], None]] = None,
    eval_env: Optional[RLEnv] = None,
    initial_state: Optional[DistillationState] = None,
) -> DistillationTrainResult:
    """Train a student network by distillation from a frozen teacher.

    Args:
        env: Training environment.
        teacher: Frozen teacher network. Will be set to eval (deterministic) mode.
        student: Student network to train.
        config: Training configuration. If None, uses default_distillation_config().
        total_steps: Override config.distillation.total_steps.
        seed: Override config.seed.
        log_fn: Called with (metrics_dict, step) after each distillation step.
        video_fn: Called with VideoData after rendering eval episodes.
        checkpoint_fn: Called with (distillation_state, step) at checkpoint intervals.
        eval_env: Environment for evaluation rollouts. If None, uses env.
        initial_state: Resume training from an existing DistillationState.
                       If None, creates a new DistillationState.

    Returns:
        DistillationTrainResult with final state, metrics, and eval history.
    """
    if config is None:
        config = default_distillation_config()
    if total_steps is not None:
        config = dataclasses.replace(
            config,
            distillation=dataclasses.replace(
                config.distillation, total_steps=total_steps
            ),
        )
    if seed is not None:
        config = dataclasses.replace(config, seed=seed)

    if eval_env is None:
        eval_env = env

    # Teacher runs in deterministic (eval) mode so raw_actions = mu_teacher.
    teacher.eval()

    if initial_state is None:
        distillation_state = new_distillation_state(
            env,
            teacher,
            student,
            config.distillation.n_envs,
            config.seed,
            config.distillation.learning_rate,
            config.distillation.gradient_clipping,
            config.distillation.weight_decay,
        )
    else:
        distillation_state = initial_state

    # JIT compile inner functions.
    # static_argnums: env (0), n_envs (3), rollout_length (4), n_epochs (5),
    # n_minibatches (6), logging_level (7), logging_percentiles (8).
    # teacher (1) is an NNX module — handled dynamically by nnx.jit, not static.
    distillation_step_jit = nnx.jit(
        distillation_step, static_argnums=(0, 3, 4, 5, 6, 7, 8)
    )
    eval_rollout_jit = nnx.jit(rollout.eval_rollout, static_argnums=(0, 2, 3, 5))
    eval_rollout_render_jit = nnx.jit(
        rollout.eval_rollout_for_render_scan, static_argnums=(0, 2)
    )

    eval_history: list[dict[str, Any]] = []
    last_eval_step = -config.eval.every_steps  # ensure eval at step 0
    last_video_step = -config.video.every_steps
    last_checkpoint_step = -config.checkpoint_every_steps
    metrics: dict[str, Any] = {}
    n_iterations = 0

    def run_eval(steps: int) -> dict[str, Any]:
        student.eval()
        eval_metrics = eval_rollout_jit(
            eval_env,
            student,
            config.eval.n_envs,
            config.eval.max_episode_length,
            jax.random.key(config.seed),
            config.eval.logging_percentiles,
        )
        student.train()
        return dict(eval_metrics)

    def run_video(steps: int, iteration: int) -> None:
        if video_fn is None or not hasattr(eval_env, "render"):
            return
        student.eval()
        render_key = jax.random.fold_in(jax.random.key(config.seed), iteration)
        stacked_states, final_state, episode_reward = eval_rollout_render_jit(
            eval_env, student, config.video.episode_length, render_key
        )
        trajectory = rollout.unstack_trajectory(
            stacked_states, final_state, config.video.episode_length
        )
        frames = getattr(eval_env, "render")(trajectory, **config.video.render_kwargs)
        video_data = VideoData(
            frames=np.stack(frames),
            step=steps,
            episode_reward=float(episode_reward),
            episode_length=config.video.episode_length,
        )
        video_fn(video_data)
        student.train()

    # Initial eval/video/checkpoint at step 0
    steps = int(distillation_state.steps_taken)
    if config.eval.enabled:
        eval_metrics = run_eval(steps)
        metrics.update(eval_metrics)
        eval_history.append({"step": steps, **eval_metrics})
        last_eval_step = steps
    if config.video.enabled:
        run_video(steps, n_iterations)
        last_video_step = steps
    if checkpoint_fn is not None and _should_run(
        steps, last_checkpoint_step, config.checkpoint_every_steps
    ):
        checkpoint_fn(distillation_state, steps)
        last_checkpoint_step = steps
    if log_fn is not None and metrics:
        log_fn(metrics, steps)

    # Main training loop
    while int(distillation_state.steps_taken) < config.distillation.total_steps:
        distillation_state, metrics = distillation_step_jit(
            env,
            teacher,
            distillation_state,
            config.distillation.n_envs,
            config.distillation.rollout_length,
            config.distillation.n_epochs,
            config.distillation.n_minibatches,
            config.distillation.logging_level,
            config.distillation.logging_percentiles,
        )
        n_iterations += 1
        steps = int(distillation_state.steps_taken)

        if config.eval.enabled and _should_run(
            steps, last_eval_step, config.eval.every_steps
        ):
            eval_metrics = run_eval(steps)
            metrics.update(eval_metrics)
            eval_history.append({"step": steps, **eval_metrics})
            last_eval_step = steps

        if config.video.enabled and _should_run(
            steps, last_video_step, config.video.every_steps
        ):
            run_video(steps, n_iterations)
            last_video_step = steps

        if checkpoint_fn is not None and _should_run(
            steps, last_checkpoint_step, config.checkpoint_every_steps
        ):
            checkpoint_fn(distillation_state, steps)
            last_checkpoint_step = steps

        if log_fn is not None:
            log_fn(metrics, steps)

    return DistillationTrainResult(
        training_state=distillation_state,
        final_metrics=metrics,
        eval_history=eval_history,
        total_steps=int(distillation_state.steps_taken),
        total_iterations=n_iterations,
    )
