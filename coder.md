# Prefix RFT V3 Migration Log

## Goal

Migrate prefix RFT onto the `verl 0.7.1` codebase in this worktree without editing files under `verl/`.

New code lives in the top-level package:

- `prefix_rft_v3/`

Constraints:

- Do not edit `verl/`
- If upstream logic needs changes, copy it into `prefix_rft_v3/` and modify there
- Phase 1 first: run code no-prefix baseline on `v0.7.1`
- Phase 2: migrate the `recipe/prefix_rft_v2` batch processing semantics

## Key Batch Semantics To Preserve From V2

- Dataset emits multi-prefix prompt variants per sample
- Trainer repeats by `rollout.n`, projects per-prefix prompt tensors, concatenates, and reorders
- Generation output is post-processed back into:
  - raw prompt only in `prompts`
  - prefix plus continuation in `responses`
  - rebuilt `input_ids`, `attention_mask`, `position_ids`
  - `prefix_mask`
  - `prefix_relpos`
- Advantage computation uses:
  - `uid`
  - `prefix_uid`
  - `prefix_mask`
  - empty-prefix sentinel `0000-0000-0000-0000`
- Actor loss uses prefix-aware clipping / reshaping behavior

## Work Items

### Phase 0 - Branch And Package Setup

1. Create migration branch `prefix-rft-v3-v071`
2. Create top-level package skeleton under `prefix_rft_v3/`
3. Record migration plan and progress in this file

### Phase 1 - V0.7.1 Code Baseline

4. Add `prefix_rft_v3/main.py` as the new entrypoint
5. Add `prefix_rft_v3/config/prefix_rft_trainer.yaml` aligned with `verl 0.7.1`
6. Add `prefix_rft_v3/templates.py` for missing chat templates
7. Add `prefix_rft_v3/custom_rewards/code_r1.py`
8. Add baseline launch script for code no-prefix training
9. Smoke test imports, Hydra config, and trainer initialization

### Phase 2 - Prefix Dataset And Batch Flow Migration

10. Add `prefix_rft_v3/rl_dataset.py` with multi-prefix dataset outputs
11. Add scheduler helpers under `prefix_rft_v3/scheduler/`
12. Add `prefix_rft_v3/ray_trainer.py` with:
    - repeat by rollout count
    - per-prefix projection
    - concat and deterministic reorder
    - rollout-time random reorder and inverse reorder
    - `post_process_gen_batch()`
13. Ensure reward still sees raw prompt plus prefixed response semantics

### Phase 3 - Prefix Advantage And Actor Logic

14. Add `prefix_rft_v3/core_algos.py` with prefix-aware advantage logic
15. Add `prefix_rft_v3/dp_actor.py` copied from upstream actor and modified for prefix loss behavior
16. Add `prefix_rft_v3/fsdp_workers.py` copied from upstream worker and redirected to `prefix_rft_v3.dp_actor`
17. Add custom task runner / trainer wiring so we use our copied components without touching `verl/`

### Phase 4 - Validation And Push

18. Validate baseline path
19. Validate prefix dataset shapes and batch fields
20. Validate prefix trainer batch flow
21. Commit and push branch to remote

## Validation Checklist

- `python3 -m prefix_rft_v3.main --help` works
- Hydra config resolves without legacy reward layout errors
- Custom code reward loads from `reward.custom_reward_function.*`
- Baseline run reaches trainer initialization on `v0.7.1`
- Prefix dataset emits expected `input_ids_i`, `attention_mask_i`, `position_ids_i`
- Prefix post-process reconstructs `prefix_mask` and `prefix_relpos`
- Actor update consumes migrated batch keys without editing `verl/`

## Progress

- [x] Plan finalized
- [x] Branch created: `prefix-rft-v3-v071`
- [x] Top-level package scaffolded
- [x] Phase 1 baseline path implemented
- [ ] Phase 2 batch flow migrated
- [ ] Phase 3 actor and worker migration completed
- [ ] Validation completed
- [ ] Branch pushed

## Notes

- Local phase-1 validation completed without GPU-dependent trainer startup.
- Confirmed locally:
  - python syntax for new files
  - `prefix_rft_v3` module imports
  - Hydra config composition
  - custom reward loading through `reward.custom_reward_function`
- Full trainer initialization and training smoke test should be run on a GPU machine.
- Phase 2 trainer migration is in progress in `prefix_rft_v3/ray_trainer.py`.
- Current prefix training path now:
  - expands one raw sample into multiple prefix-conditioned `raw_prompt` variants
  - keeps validation on the baseline non-prefix path
  - uses `agent_loop` for rollout
  - post-processes rollout output back into response-side prefix semantics via `post_process_gen_batch()`
  - reconstructs `prefix_mask` and `prefix_relpos`
- `prefix_rft_v3/task_runner.py` now switches to the copied trainer when `prefix_rft.enable=true`.
- Still pending after current trainer patch:
  - migrate `prefix_rft_v3/dp_actor.py`
  - wire entropy-based `reshape_func(...)` on prefix tokens
  - run GPU smoke test for the new prefix training path

## Dataset Understanding

- The current target data contract is defined by `prefix_rft_v3/rl_dataset.py`.
- For each raw data point:
  - `messages` is the original prompt in chat-message format.
  - `raw_prompt_lengths` stores the token length of the baseline prompt rendered with `add_generation_prompt=True`.
  - The dataset emits multiple prefix slots via `num_prefix`.
- For each prefix slot `i`, the dataset currently emits:
  - tensor fields:
    - `input_ids_i`
    - `attention_mask_i`
    - `position_ids_i`
  - non-tensor fields:
    - `prefix_ratio_i`
    - `prefix_uid_i`
    - `prefix_correct_i`
    - `demo_len_i`
    - `prefix_ctrl_low_i`
    - `prefix_ctrl_high_i`
- Prefix slots are built by sampling a demonstration, truncating it to a prefix, and rendering:
  - empty prefix -> baseline prompt with `add_generation_prompt=True`
  - non-empty prefix -> assistant continuation prompt with `continue_final_message=True`
- Important semantic point:
  - In the new `verl` path, these `input_ids_i` are not the rollout interface.
  - `agent_loop` consumes `raw_prompt` messages, not trainer-side token tensors.
  - Therefore the trainer must convert each prefix slot back into a `raw_prompt` variant for rollout.

## Current Implementation State

- `prefix_rft_v3/task_runner.py` now switches trainer class based on `prefix_rft.enable`:
  - `false` -> upstream `verl.trainer.ppo.ray_trainer.RayPPOTrainer`
  - `true` -> copied `prefix_rft_v3.ray_trainer.RayPPOTrainer`
- `prefix_rft_v3/ray_trainer.py` now contains prefix-specific helper code:
  - `build_prefilled_raw_prompt(...)`
  - `decode_prefix_text(...)`
  - `build_prefix_slot_batches(...)`
  - `compute_prefix_reorder_indices(...)`
  - `reorder_batch_randomly(...)`
  - `post_process_gen_batch(...)`
- The trainer currently attempts this Phase-2 flow when `prefix_rft.enable=true`:
  - build one rollout request per prefix slot using `raw_prompt`
  - expand by `rollout.n`
  - deterministic reorder
  - random reorder for balanced rollout
  - async rollout via `agent_loop`
  - post-process output into:
    - baseline `prompts`
    - hybrid `responses = prefix + continuation`
    - rebuilt `input_ids`, `attention_mask`, `position_ids`
    - `prefix_mask`
    - `prefix_relpos`
- Validation is intentionally still baseline / non-prefix only.

## Important Realizations

- Prefix-RFT in the new stack is not a "token-input rollout" problem.
- It is a "raw-prompt rollout + response-side reconstruction" problem.
- `post_process_gen_batch(...)` is still necessary because prefix tokens must become part of the response-side training object.
- Advantage grouping stays on original `uid`, not `prefix_uid`.
- The entropy-based masking logic is not reward-side and not entropy-loss-side.
- It belongs in actor update, where prefix tokens will have token-level `advantages` reshaped/masked based on entropy.

## Current Known Gaps / Risks

- The trainer code is only partially migrated.
- The most important unresolved area is the final training batch assembly in `prefix_rft_v3/ray_trainer.py` after rollout:
  - the prefix path currently routes training through `gen_batch_output` directly
  - this avoids some tensor key conflicts from the dataset
  - but it still needs a careful review to ensure all non-tensor fields needed downstream are preserved exactly once
- The current dataset still emits top-level baseline fields like:
  - `input_ids`
  - `attention_mask`
  - `position_ids`
  - `prefix_ratio`
  - `prefix_uid`
  - `prefix_correct`
- Those top-level fields are convenient for compatibility, but they are also a source of collision risk when unioning with rollout outputs.
- A likely cleanup option for the next session is:
  - reduce the prefix dataset's rollout-facing surface
  - keep only slot-specific fields plus `raw_prompt`
  - avoid carrying baseline tensor aliases that are no longer used by `agent_loop`

## What Was Verified Locally

- `python -m py_compile prefix_rft_v3/ray_trainer.py prefix_rft_v3/task_runner.py`
- `python -c "from prefix_rft_v3.ray_trainer import RayPPOTrainer; print(RayPPOTrainer.__name__)"`
- These checks passed on the devbox.

## Recommended Restart Point

- Start from `prefix_rft_v3/ray_trainer.py`.
- First inspect the current prefix-specific methods around:
  - `build_prefix_slot_batches(...)`
  - `_build_prefix_training_batch(...)`
  - `_generate_prefix_sequences(...)`
  - `_finalize_prefix_training_batch(...)`
  - the `fit()` generation block under `if self.prefix_rft_enabled:`
- The next concrete goal is to stabilize trainer-side batch assembly before moving to actor migration.
- Only after that should the next session copy and patch:
  - `prefix_rft_v3/dp_actor.py`
  - and, if needed, `prefix_rft_v3/fsdp_workers.py`

## Guidance For Next Session

- Do not touch validation prefix behavior.
- Continue to treat `agent_loop` as the only rollout engine.
- Do not re-open the question of token-input rollout.
- If there is uncertainty, prefer preserving batch alignment over preserving old dataset compatibility aliases.

## Session Update - 2026-03-28

- Reviewed the current copied prefix path and the upstream `verl` contracts in:
  - `prefix_rft_v3/ray_trainer.py`
  - `prefix_rft_v3/rl_dataset.py`
  - `prefix_rft_v3/task_runner.py`
  - `verl/protocol.py`
  - `verl/trainer/ppo/ray_trainer.py`
  - `verl/experimental/agent_loop/agent_loop.py`
  - `verl/experimental/reward_loop/reward_loop.py`
- Confirmed the key rollout contract is still correct:
  - prefix training is a `raw_prompt` rollout path, not a direct `input_ids_i` rollout path
  - each prefix slot must be decoded back into assistant-prefilled `raw_prompt`
  - the prefix path should expand by `rollout.n`, reorder into sample-major prefix-major rollout-major layout, optionally randomize for rollout load balancing, then inverse-reorder after generation
- Confirmed `post_process_gen_batch()` is the right reconstruction point for V2 semantics:
  - final `prompts` should contain only the raw prompt
  - final `responses` should contain prefix plus continuation
  - final `input_ids`, `attention_mask`, and `position_ids` must be rebuilt from those reconstructed prompt/response tensors
  - `prefix_mask` and `prefix_relpos` belong to the finalized response-side training batch
- The current highest-risk gap in Phase 2 is batch ownership after rollout, not prefix decoding itself:
  - upstream PPO keeps the repeated original batch and `union(...)`s rollout outputs into it
  - the copied prefix path currently replaces the training batch with finalized `gen_batch_output`
  - therefore the finalized prefix batch must be treated as fully self-contained
- Fields that must remain aligned through repeat, reorder, and finalize in the prefix path:
  - `uid` for GRPO grouping and PrefixGrouper-compatible balancing
  - reward extra-info fields whenever `agent_loop` already returned `rm_scores` and `reward_extra_keys`
  - `meta_info["temperature"]` for downstream actor/ref log-prob computation
  - `raw_prompt` is not required by the current baseline config because `reward.reward_model.enable=false`, but it will be required again if colocated reward-model scoring is enabled
- Config state checked during this review:
  - `prefix_rft_v3/config/prefix_rft_trainer.yaml` sets `return_raw_chat: True`
  - `prefix_rft_v3/config/prefix_rft_trainer.yaml` keeps `reward.reward_model.enable: False`
  - `prefix_rft_v3/task_runner.py` switches to the copied dataset and copied trainer only when `prefix_rft.enable=true`
- There is no repo-local `recipe/prefix_rft_v2` copy in this worktree, so further migration work should continue from the copied `prefix_rft_v3` implementation and upstream `verl` behavior.
- Next concrete edit target remains `prefix_rft_v3/ray_trainer.py`, especially:
  - `_generate_prefix_sequences(...)`
  - `_finalize_prefix_training_batch(...)`
  - preservation rules for non-tensor fields and meta info after rollout

## Implementation Update - Minimal Run-First Patch

- Adjusted course away from a "perfect migration" and toward the minimum trainer fixes needed to get baseline and prefix-RFT running.
- Implemented the first stabilization patch in `prefix_rft_v3/ray_trainer.py`:
  - `post_process_gen_batch(...)` now rebuilds finalized response-side tensors while preserving rollout outputs that downstream PPO still needs:
    - `response_mask`
    - `rm_scores` when returned by `agent_loop`
    - `rollout_log_probs` when returned by `agent_loop`
  - prefix tokens are prepended into finalized `responses`, but their loss mask is zeroed in `response_mask`
  - `raw_prompt` and the repeated non-tensor fields are now kept on the finalized prefix batch instead of being dropped
  - `_generate_prefix_sequences(...)` now re-attaches preserved non-tensor fields after rollout and asserts finalized batch size matches the repeated rollout batch size
  - `_finalize_prefix_training_batch(...)` now restores `meta_info["temperature"]`, `meta_info["global_steps"]`, and `meta_info["global_step"]`
- Local non-GPU checks completed:
  - `python -m py_compile prefix_rft_v3/ray_trainer.py prefix_rft_v3/rl_dataset.py prefix_rft_v3/task_runner.py`
  - synthetic Python sanity check for `post_process_gen_batch(...)` covering shape alignment of:
    - `responses`
    - `response_mask`
    - `prefix_mask`
    - `rm_scores`
    - `rollout_log_probs`
- Important caveat:
  - this is still a trainer-only stabilization patch intended to unblock runs quickly
  - actor-side prefix-specific advantage/entropy reshaping is still not migrated in this session

## Implementation Update - Minimal Actor Migration

- Prefix actor migration has now been started with a run-first scope, using the v2 actor logic as reference rather than trying to fork all upstream worker code.
- Trainer-side semantic correction applied first in `prefix_rft_v3/ray_trainer.py`:
  - prefix tokens are confirmed to be part of `responses`
  - finalized `response_mask` now keeps prefix tokens active instead of zeroing them out
  - `prefix_mask` is still emitted separately so actor-side logic can distinguish prefix tokens from ordinary continuation tokens
  - batch-level `meta_info["prefix_ctrl_low"]` and `meta_info["prefix_ctrl_high"]` are now attached for actor reshaping
- Added `prefix_rft_v3/dp_actor.py`:
  - subclasses upstream `verl.workers.actor.dp_actor.DataParallelPPOActor`
  - ports a minimal version of v2 `reshape_func(...)`
  - currently supports the main prefix reshapers needed for smoke testing:
    - `identity`
    - `const`
    - `entropy`
    - `entropy_low`
    - `entropy_op`
    - `position`
    - `random`
    - `length_on_pos`
    - `length_all_pos`
    - `length`
  - custom actor `update_policy(...)` now:
    - consumes `prefix_mask` and `prefix_relpos`
    - applies prefix-aware advantage reshaping before PPO loss
    - supports `replace_old_log_prob`
    - applies entropy regularization on non-prefix response tokens when `entropy_mode=response_only`
    - otherwise keeps upstream PPO loss machinery (`get_policy_loss_fn(...)`) intact
- Added `prefix_rft_v3/fsdp_workers.py`:
  - subclasses upstream FSDP async worker
  - injects minimal prefix actor knobs from `prefix_rft` config into actor/ref config
  - swaps upstream actor/ref instances to `prefix_rft_v3.dp_actor.DataParallelPPOActor`
- Updated `prefix_rft_v3/task_runner.py`:
  - when `prefix_rft.enable=true` and strategy is FSDP/FSDP2, the task runner now uses `prefix_rft_v3.fsdp_workers.AsyncActorRolloutRefWorker`
- Updated `prefix_rft_v3/config/prefix_rft_trainer.yaml` with minimal actor-facing prefix defaults:
  - `off_adv_reshaper: entropy`
  - `off_ent_mask_ratio_ctrl_type: const`
  - `off_ent_mask_ratio_ctrl.kwargs.init: 0.8`
  - `entropy_mode: response_only`
  - `replace_old_log_prob: False`
- Local checks completed after this actor patch:
  - `python -m py_compile prefix_rft_v3/ray_trainer.py prefix_rft_v3/rl_dataset.py prefix_rft_v3/task_runner.py prefix_rft_v3/main.py prefix_rft_v3/dp_actor.py prefix_rft_v3/fsdp_workers.py`
  - Hydra composition check for prefix mode with actor-specific prefix overrides
- Remaining caveat:
  - this is still a minimal migration on top of upstream PPO loss
  - v2's custom off-policy clip logic from `recipe/prefix_rft_v2/core_algos.py` is not ported yet
  - if smoke training still diverges semantically after actor updates, that custom loss path is the next likely place to migrate

## Next Build Plan - Baseline + Prefix-RFT Bring-Up

- Immediate goal for the next build session is no longer "more migration" in the abstract; it is to make both baseline and prefix-RFT smoke runs start successfully and reach actor update.
- The current highest-priority runtime blocker is dataset initialization multiprocessing inside `prefix_rft_v3/rl_dataset.py`:
  - removing `filter_overlong_prompts` is not sufficient
  - `datasets.Dataset.map(..., num_proc=...)` and `filter(..., num_proc=...)` still trigger `multiprocess`/`dill` serialization
  - the observed failure is `TypeError: cannot pickle 'ConfigModuleInstance' object`
  - therefore the whole prefix dataset initialization path must be treated as single-process for smoke unless proved otherwise

### Runtime Facts To Preserve

- Prefix controller/scheduler code itself is already present in `prefix_rft_v3/scheduler/global_step.py` and `prefix_rft_v3/rl_dataset.py`.
- Prefix ratio sampling already calls the controller with `global_step`, so step-aware decay/rise semantics exist in principle.
- However, the trainer does not yet keep `self.train_dataloader.dataset.global_step` synchronized with `self.global_steps`, so controller decay over training steps is not fully live yet.
- Prefix metrics partially exist in v3 (`actor/prefix_ratio`, `actor/demo_len`, `actor/prefix_low`, `actor/prefix_high`), but v2-style observability is still incomplete.
- The new reward implementation at `prefix_rft_v3/custom_rewards/code_r1_v2.py` should be treated as the preferred code reward path for upcoming runs.

### Execution Order For The Next Build Session

1. De-risk prefix dataset startup first.
   - Update `prefix_rft_v3/rl_dataset.py` so all HF datasets preprocessing uses a single-process path for smoke.
   - This includes every `map(..., num_proc=...)` and `filter(..., num_proc=...)` call, not just the overlong-prompt filter.
   - Favor correctness and startup stability over preprocessing speed.

2. Switch custom module loading to package paths.
   - In `prefix_rft_v3/task_runner.py`, change the custom dataset path from file-style loading to `pkg://prefix_rft_v3.rl_dataset`.
   - Also switch the default reward path to `pkg://prefix_rft_v3.custom_rewards.code_r1_v2`.
   - Goal: reduce serialization/import edge cases and align with stable module import behavior.

3. Finish wiring the prefix controller to live training steps.
   - In `prefix_rft_v3/ray_trainer.py`, mirror v2 behavior by setting `self.train_dataloader.dataset.global_step = self.global_steps` after checkpoint load and after each training step.
   - This is required for cosine/linear/delay controllers to actually change prefix length over time.

4. Restore the minimum useful prefix metrics for smoke diagnosis.
   - Add v2-style metrics that help verify prefix tokens are truly participating in training, especially:
     - `actor/num_response_tokens`
     - `actor/num_prefix_tokens`
     - `actor/off_ratio`
   - Keep existing `actor/prefix_ratio`, `actor/prefix_low`, `actor/prefix_high`, and `actor/demo_len`.

5. Verify the baseline path again after the dataset/reward-path cleanup.
   - Baseline should continue to run through the shared v3 entrypoint with `prefix_rft.enable=false`.
   - The goal is to confirm that prefix-specific startup fixes do not regress the baseline smoke path.

6. Re-run prefix-RFT smoke and focus on first-step survivability.
   - Success criteria for the first prefix smoke are:
     - dataset initialization no longer crashes on pickling
     - rollout finishes
     - reward extraction works with the finalized prefix batch
     - first `update_actor` completes
     - prefix metrics are visible
     - controller-driven prefix settings change with training step when configured to decay

7. Only after smoke runs are stable, port the remaining v2 custom off-policy PPO loss semantics.
   - The main remaining semantic gap is still the v2 custom `compute_policy_loss(...)` path from `recipe/prefix_rft_v2/core_algos.py`.
   - That later migration should cover:
     - `prefix_mask`-aware clipping
     - off-policy clip bounds (`clip_ratio_low_off`, `clip_ratio_high_off`)
     - `enable_clip`
     - prefix-specific clip metrics
   - This is important for fidelity to v2, but it is intentionally sequenced after startup stability.

### Acceptance Criteria For The Next Session

- Baseline smoke starts and reaches training update without dataset startup regressions.
- Prefix-RFT smoke starts without HF dataset pickling failures.
- Prefix-RFT reaches first actor update with `prefix_mask`, `prefix_relpos`, `prefix_ctrl_low`, and `prefix_ctrl_high` flowing through.
- Prefix controller behavior is live across training steps, not frozen at dataset initialization.
- New reward path uses `code_r1_v2.py` by default for code tasks.
- If remaining failures persist after these steps, the next debugging target should be the v2 custom off-policy policy-loss port rather than more dataset/trainer scaffolding.

## Implementation Update - Startup And Metrics Pass

- Implemented the dataset startup stabilization in `prefix_rft_v3/rl_dataset.py`.
  - Added a dedicated `dataset_num_proc` knob with default `1`.
  - All HF `Dataset.map(...)` and `Dataset.filter(...)` calls now go through single-process helpers by default.
  - This removes the smoke-path dependency on `multiprocess` pickling for closures that capture tokenizer/config state.
- Switched custom module loading to package imports.
  - `prefix_rft_v3/task_runner.py` now sets the custom dataset path to `pkg://prefix_rft_v3.rl_dataset`.
  - `prefix_rft_v3/config/prefix_rft_trainer.yaml` now points the default reward function to `pkg://prefix_rft_v3.custom_rewards.code_r1_v2`.
  - Both launch scripts under `prefix-rft-jobs/` now override reward loading with the same package path.
  - Added `prefix_rft_v3/custom_rewards/__init__.py` so the reward module resolves as a normal package.
- Wired live prefix controller step updates in `prefix_rft_v3/ray_trainer.py`.
  - Added `_sync_train_dataset_global_step()`.
  - The trainer now syncs dataset `global_step` after checkpoint load, after training-step initialization, and after each completed step.
- Restored prefix smoke metrics in `prefix_rft_v3/ray_trainer.py`.
  - Added:
    - `actor/num_response_tokens`
    - `actor/num_prefix_tokens`
    - `actor/off_ratio`
  - Existing prefix observability remains in place:
    - `actor/prefix_ratio`
    - `actor/prefix_low`
    - `actor/prefix_high`
    - `actor/demo_len`
- Remaining work after this pass is mainly runtime validation on GPU.
  - The next check should be a real prefix smoke run to confirm the HF dataset pickling failure is gone and first `update_actor` completes.
