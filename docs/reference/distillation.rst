Distillation
============

:func:`~nnx_ppo.algorithms.distillation.train_distillation` trains a
*student* network to imitate a frozen *teacher* (Policy Distillation,
Rusu et al. 2015). Both are ordinary actor-critic networks of the kind
:func:`~nnx_ppo.algorithms.ppo.train_ppo` takes. A typical use is to
distil a policy trained with privileged information into one that has
to make do without it.

The algorithm
-------------

Each iteration:

1. Roll out the environment with the **student's** actions, so training
   happens on the student's own state distribution.
2. Run the teacher alongside on the same observations, in eval
   (deterministic) mode. Its samplers then emit the action *mean*
   rather than a sample.
3. Replay the student over the rollout and minimise the negative
   log-likelihood of the teacher's mean under the student's action
   distribution, ``-log p_student(mu_teacher | obs)``. Up to the
   teacher's (constant) entropy this is ``KL(teacher || student)``.
   The student's own regularisation losses (entropy bonus, KL terms,
   auxiliary losses) are added as usual.
4. Repeat for ``n_epochs * n_minibatches`` gradient updates, as in PPO.

The student's critic is carried along but not trained: the loss uses
only the action distribution.

How the target reaches the student
----------------------------------

Step 3 reuses the machinery PPO uses to replay a rollout. During a
rollout every module may emit ``rollout_extras``; a sampler emits the raw
(pre-squashing) action it produced. In the loss replay those extras are
passed back in, and a sampler that receives one computes the
log-likelihood of that stored action instead of drawing a new one.
Distillation simply passes the **teacher's** extras in place of the
student's, so each student sampler scores the teacher's mean.

Containers route extras by *position* -- ``Sequential`` by layer index,
``PPOAdapter`` / ``Map`` / ``Concat`` by key -- so by default this only
works when the teacher's and student's ``rollout_extras`` trees are
isomorphic: the same skeleton, with samplers at matching positions. The
easiest way to satisfy that is to build both with the same factory.
Their *state* trees may differ freely; each network carries its own.

Teachers and students with different skeletons
----------------------------------------------

Often the interesting student is *not* shaped like its teacher -- for
example an undelayed teacher distilled into a student with an extra
observation-delay layer in front of its actor. Pass a ``target_fn``::

    def target_fn(teacher_extras, student_extras):
        # return a tree shaped like student_extras
        ...

    train_distillation(env, teacher, student, config, target_fn=target_fn)

It is called once per rollout step, inside the jitted rollout, with both
networks' extras for that step, and its result is stored as the target.
The usual implementation takes the student's own extras and replaces the
sampler leaves with the teacher's. Leaves it keeps from the student are
fed back to the student unchanged, which is what they would have been in
PPO.

The mapping is then the caller's responsibility. A target routed to the
wrong position trains the student against the wrong quantity without
any error, so a ``target_fn`` should assert what it relies on -- for
instance that it found exactly one sampler leaf in each tree and that
their shapes agree. ``target_fn`` is a static argument of the jitted
step, so pass a module-level function rather than a new lambda each
call.

Without ``target_fn`` the teacher's extras are used as they are, which
is the original behaviour.

Checkpointing and resuming
--------------------------

``train_distillation`` takes the same ``checkpoint_fn``, ``stop_fn`` and
``initial_eval`` arguments as ``train_ppo``, and
:func:`~nnx_ppo.algorithms.checkpointing.make_checkpoint_fn` accepts a
``DistillationState``. See :doc:`checkpointing`, in particular
"Resuming an interrupted run" and "Distillation checkpoints".
