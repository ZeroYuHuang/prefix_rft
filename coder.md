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

## Implementation Update - Validation Dataset Split Back To Upstream

- Adjusted `prefix_rft_v3/task_runner.py` so prefix mode no longer forces both train and val onto `PrefixRFTDataset`.
- Current dataset split behavior is now:
  - train dataset uses `pkg://prefix_rft_v3.rl_dataset:PrefixRFTDataset`
  - val dataset falls back to upstream `verl.utils.dataset.rl_dataset.RLHFDataset`
- `return_raw_chat=True` is still preserved for both train and val so agent-loop rollout keeps working.
- Motivation:
  - validation never consumes prefix-specific slot tensors
  - using `PrefixRFTDataset` for val was emitting top-level `input_ids` / `attention_mask` / `position_ids`
  - those prompt-side tensors conflicted with rollout output tensors during `_validate()` union, causing:
    - `AssertionError: input_ids in tensor_dict1 and tensor_dict2 are not the same object`
- Expected impact:
  - validation path is now closer to upstream PPO behavior
  - prefix-specific batch fields remain isolated to the training dataset where they are actually used

## Implementation Update - Add Dedicated Prefix Verify Script

- Added `prefix-rft-jobs/code-qwen3-0p6b-prefix-rft-v3-verify.sh`.
- This wrapper keeps the formal `code-qwen3-0p6b-prefix-rft-v3.sh` launch path unchanged and only overrides a small verification profile:
  - `trainer.total_epochs=1`
  - `trainer.total_training_steps=5`
  - `trainer.val_before_train=true`
  - `trainer.test_freq=1000`
  - `trainer.save_freq=5`
  - `data.train_max_samples=64`
  - `data.val_max_samples=64`
  - `data.val_batch_size=64`
- Purpose:
  - validate val-before-train
  - validate prefix rollout/reorder path
  - validate real reward execution
  - validate first actor update and one save cycle with minimal cost

## Implementation Update - Add Qwen3-1.7B Prefix Launchers

- Added `prefix-rft-jobs/code-qwen3-1p7b-prefix-rft-v3.sh`.
- Added `prefix-rft-jobs/code-qwen3-1p7b-prefix-rft-v3-verify.sh`.
- The `1p7b` launcher keeps the same prefix-RFT recipe as the working `0p6b` script, but starts from a more conservative resource profile:
  - `MODEL_PATH=/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/models/Qwen3-1.7B-Base`
  - `data.train_batch_size=8`
  - `data.val_batch_size=128`
  - `actor_rollout_ref.actor.ppo_mini_batch_size=32`
  - `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16`
  - `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384`
  - `actor_rollout_ref.rollout.max_num_batched_tokens=24576`
  - `actor_rollout_ref.rollout.gpu_memory_utilization=0.55`
- The `1p7b` verify wrapper mirrors the `0p6b` verify pattern and only overrides a short 5-step validation profile.

## Implementation Update - Dataset Alignment With Upstream RLHFDataset

- Refactored `prefix_rft_v3/rl_dataset.py` so `PrefixRFTDataset` now subclasses upstream `RLHFDataset` instead of reimplementing the whole loading stack.
- The prefix dataset now reuses upstream behavior for:
  - file download/cache
  - parquet loading and concatenation
  - `max_samples` selection
  - prompt-length filtering
  - resume / `__getstate__` behavior inherited from upstream
  - multimodal-aware `_build_messages()` path
- Prefix-specific behavior remains in the subclass:
  - demonstration column normalization
  - demo dropout / optional demo-only filtering
  - prefix controller state (`uid`, `avg_rewards`, `prefix_ratios`, `training_metrics`, `global_step`)
  - prefix-aware `__getitem__` output contract used by `prefix_rft_v3/ray_trainer.py`
- To preserve the previous prefix config behavior, the subclass overrides `maybe_filter_out_long_prompts()` and temporarily uses `filter_overlong_prompts_length` for upstream filtering before restoring the actual training `max_prompt_length`.
- `_build_messages()` now delegates to upstream on a deep-copied example so we reuse upstream placeholder handling without mutating stored samples.
- Local validation on real data succeeded for the new dataset path:
  - instantiated `PrefixRFTDataset` on the real train parquet
  - validated both `filter_overlong_prompts_workers=1` and `16`
  - fetched a sample and confirmed prefix fields such as `prefix_ratio_0`, `prefix_uid_0`, and `raw_prompt_lengths` are produced
- Expected impact:
  - prefix data loading should now behave much closer to baseline/upstream
  - any remaining failures are more likely to be true prefix runtime issues after dataset construction, rather than custom dataset ingestion drift

## Finalized Recipe Decisions And Detailed Work Orders

This section supersedes the earlier idea of migrating v2 prefix-aware advantage estimators.

### Final Recipe Decisions

- The final training recipe is:
  - build a hybrid trajectory where the prefix tokens are part of the response sequence
  - compute reward on the full hybrid trajectory
  - compute advantage with ordinary GRPO grouped by `uid`
  - let prefix tokens and on-policy continuation share the same advantage before actor-side reshaping
  - apply prefix-specific masking / reshaping only in the actor update path
- Therefore we explicitly do **not** plan to migrate the old v2 prefix-aware advantage branches such as:
  - `recipe/prefix_rft_v2/ray_trainer.py:160`
  - `recipe/prefix_rft_v2/core_algos.py:162`
  - `recipe/prefix_rft_v2/core_algos.py:262`
- The v2 custom policy loss is **optional for recipe semantics but recommended for extensibility**.
  - It does not change the shared-advantage recipe.
  - It adds an optional prefix-specific PPO clipping path and extra diagnostics.

### P0-1 Freeze Shared-Advantage Semantics

- Goal:
  - keep ordinary GRPO semantics so prefix tokens and continuation tokens share the same advantage before reshape.
- Implementation:
  - keep `prefix_rft_v3/ray_trainer.py` on the ordinary GRPO path.
  - do not add any new `prefix_uid`-aware advantage estimator to the active recipe.
  - add a short inline note in the relevant trainer logic so future edits do not accidentally revive the deprecated v2 path.
- References:
  - `prefix_rft_v3/ray_trainer.py:182`
  - `recipe/prefix_rft_v2/ray_trainer.py:160`
  - `recipe/prefix_rft_v2/core_algos.py:162`
  - `recipe/prefix_rft_v2/core_algos.py:262`
- Acceptance:
  - prefix training still uses ordinary `GRPO(uid)` advantage computation.
  - no new active config path depends on `prefix_uid` during advantage computation.

### P0-2 Finalize Hybrid-Trajectory Semantics In Trainer

- Goal:
  - ensure the prefix is treated as response content rather than prompt-only context.
- Implementation:
  - verify and keep the current finalized batch path where prefix tokens are merged into `responses`.
  - keep `response_mask` equal to `1` on prefix tokens.
  - keep `prefix_mask` and `prefix_relpos` aligned with the finalized response layout.
  - add or tighten assertions around the finalized prefix batch so semantic regressions fail early.
- References:
  - `prefix_rft_v3/ray_trainer.py:377`
  - `prefix_rft_v3/ray_trainer.py:482`
  - `prefix_rft_v3/ray_trainer.py:795`
  - `recipe/prefix_rft_v2/ray_trainer.py:1115`
  - `recipe/prefix_rft_v2/ray_trainer.py:1153`
- Acceptance:
  - non-empty prefix samples have `prefix_mask.sum() > 0`.
  - prefix tokens are included in `response_mask`.
  - reward and GRPO operate on the full hybrid trajectory.

### P0-3 Finalize Actor-Side Reshape And Mask Path

- Goal:
  - make the actor path match the intended recipe: shared advantage first, then prefix-aware reshape / mask.
- Implementation:
  - keep `reshape_func(...)` as the only place where prefix tokens are specially modified before PPO loss.
  - audit that `prefix_mask`, `prefix_relpos`, `prefix_ctrl_low`, `prefix_ctrl_high`, and `replace_old_log_prob` flow through the current actor path.
  - keep entropy masking behavior controlled by `entropy_mode=response_only`.
- References:
  - `prefix_rft_v3/dp_actor.py:67`
  - `prefix_rft_v3/dp_actor.py:327`
  - `recipe/prefix_rft_v2/dp_actor.py:720`
  - `recipe/prefix_rft_v2/dp_actor.py:735`
- Acceptance:
  - prefix tokens and continuation tokens enter actor update with the same base advantage.
  - prefix-specific change only happens through reshape / masking logic.
  - `replace_old_log_prob` continues to work when enabled.

### P0-4 Implement Prefix-Specific PPO Clipping Mode For Extensibility

- Goal:
  - port the useful part of the v2 custom policy loss as a prefix-only, optional PPO clipping mode.
- Why this is still worth doing:
  - it does not change the shared-advantage recipe.
  - it adds a clean extension point for future experiments on prefix-token update strength.
  - it adds prefix-only diagnostics that upstream loss does not expose.
- Implementation:
  - create `prefix_rft_v3/core_algos.py`.
  - copy upstream `vanilla` PPO loss from `verl/trainer/ppo/core_algos.py` as the base implementation.
  - register a new prefix loss mode, e.g. `prefix_v2`.
  - add the minimal v2-only extensions:

## Implementation Update - Formal 0.6B Experiment Scripts Finalized

- Updated `prefix-rft-jobs/code-qwen3-0p6b-prefix-rft-v3.sh` into the formal long-context PRFT launcher.
- Updated `prefix-rft-jobs/code-qwen3-0p6b-rft-baseline-v3.sh` into the matching long-context baseline launcher.

### Final baseline semantics

- Baseline uses pure on-policy sampling with:
  - `actor_rollout_ref.rollout.n=8`
  - no prefix dataset path enabled

### Final PRFT semantics

- PRFT uses fixed 8-slot sampling per sample with:
  - `prefix_rft.num_prefix=8`
  - `prefix_rft.num_empty_prefix=7`
  - `actor_rollout_ref.rollout.n=1`
- This matches the intended comparison shape:
  - baseline: 8 pure on-policy rollouts
  - prft: 7 empty-prefix rollouts + 1 prefixed rollout

### Final long-context profile shared by both scripts

- `data.max_prompt_length=8192`
- `data.max_response_length=8192`
- `actor_rollout_ref.rollout.max_model_len=16384`
- `data.train_batch_size=128`
- `data.val_batch_size=64`
- `actor_rollout_ref.actor.use_dynamic_bsz=true`
- `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true`
- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536`
- `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536`
- `actor_rollout_ref.rollout.max_num_batched_tokens=131072`

### Final PRFT controller schedule

- `prefix_high_ctrl_type=const`
- `prefix_high_ctrl.kwargs.init=0.95`
- `prefix_low_ctrl_type=cosine_decay`
- `prefix_low_ctrl.kwargs.init=0.95`
- `prefix_low_ctrl.kwargs.target=0.5`
- `prefix_low_ctrl.kwargs.n_steps` is now exposed as `PREFIX_LOW_DECAY_STEPS`, defaulting to `TOTAL_TRAINING_STEPS`.

## Implementation Update - Formal 1.7B Experiment Scripts Prepared

- Rebuilt `prefix-rft-jobs/code-qwen3-1p7b-prefix-rft-v3.sh` to match the finalized `0.6B` formal PRFT structure.
- Added `prefix-rft-jobs/code-qwen3-1p7b-rft-baseline-v3.sh` as the matching formal baseline launcher.
- Added `prefix-rft-jobs/code-qwen3-1p7b-rft-baseline-v3-verify.sh`.
- Updated `prefix-rft-jobs/code-qwen3-1p7b-prefix-rft-v3-verify.sh` so the verify path stays lightweight by overriding to `3072/3072` prompt/response limits for short smoke runs.
- The formal `1.7B` scripts now follow the same semantics as `0.6B`:
  - baseline: `rollout.n=8`
  - prft: `num_prefix=8`, `num_empty_prefix=7`, `rollout.n=1`
  - shared long-context defaults: `8192/8192` with `max_model_len=16384`
  - shared large-token dynamic-batch profile: `ppo_max_token_len_per_gpu=65536`, `log_prob_max_token_len_per_gpu=65536`, `rollout.max_num_batched_tokens=131072`
    - `prefix_mask`
    - `clip_ratio_low_off`
    - `clip_ratio_high_off`
    - `enable_clip`
    - prefix clip metrics
  - do **not** add any prefix-aware advantage logic here.
- References:
  - `verl/trainer/ppo/core_algos.py:1276`
  - `recipe/prefix_rft_v2/core_algos.py:544`
  - `recipe/prefix_rft_v2/dp_actor.py:739`
- Acceptance:
  - prefix runs can opt into `policy_loss.loss_mode=prefix_v2`.
  - metrics include:
    - `actor/pg_clipfrac_prefix`
    - `actor/pg_clipfrac_lower_prefix`
  - baseline remains on upstream `vanilla` loss.

### P0-5 Wire Custom Policy Loss Into Prefix Actor And Config

- Goal:
  - make the new prefix-specific policy loss callable from the current prefix worker path without affecting baseline.
- Implementation:
  - import and register `prefix_rft_v3.core_algos` before actor updates begin.
  - update `prefix_rft_v3/dp_actor.py` to pass `prefix_mask` into the prefix loss mode when active.
  - wire prefix-only config knobs for:
    - `policy_loss.loss_mode=prefix_v2`
    - `clip_ratio_low_off`
    - `clip_ratio_high_off`
    - `enable_clip`
  - inject those knobs through the existing prefix worker/config wiring rather than modifying upstream baseline config behavior.
- References:
  - `prefix_rft_v3/main.py:10`
  - `prefix_rft_v3/dp_actor.py:344`
  - `prefix_rft_v3/fsdp_workers.py`
  - `prefix_rft_v3/config/prefix_rft_trainer.yaml`
  - `verl/workers/config/actor.py:71`
  - `verl/workers/actor/dp_actor.py:603`
- Acceptance:
  - prefix jobs can enable the custom loss mode by config.
  - baseline jobs keep the upstream actor loss path.
  - missing prefix-only knobs fall back to safe defaults.

### P1-6 Restore And Extend Prefix Diagnostics

- Goal:
  - make the prefix path debuggable enough for real runs, not only smoke runs.
- Implementation:
  - keep current metrics:
    - `actor/num_response_tokens`
    - `actor/num_prefix_tokens`
    - `actor/off_ratio`
    - `actor/prefix_ratio`
    - `actor/prefix_low`
    - `actor/prefix_high`
    - `actor/demo_len`
  - add the prefix clipping metrics from the custom policy loss path.
  - add assertions or guarded logs for obviously invalid states.
- References:
  - `prefix_rft_v3/ray_trainer.py:832`
  - `prefix_rft_v3/ray_trainer.py:863`
  - `recipe/prefix_rft_v2/dp_actor.py:785`
- Acceptance:
  - metrics clearly show whether prefix tokens are present and how strongly they are being clipped.
  - invalid prefix states fail fast instead of silently producing bad training.

### P1-7 Prefix Script Finalization

- Goal:
  - make the launch script explicitly reflect the finalized recipe and the optional prefix PPO extension.
- Implementation:
  - keep the prefix job on the shared-advantage GRPO path.
  - add explicit prefix-only overrides for the optional custom policy loss mode and its knobs.
  - keep baseline script free of all prefix-specific loss settings.
- References:
  - `prefix-rft-jobs/code-qwen3-0p6b-prefix-rft-v3.sh`
  - `prefix-rft-jobs/code-qwen3-0p6b-rft-baseline-v3.sh`
- Acceptance:
  - prefix and baseline scripts only differ on intended prefix-specific knobs.
  - there is no accidental route from baseline into prefix-only loss logic.

### P1-8 Real-Data Readiness Gate

- Goal:
  - define the minimum runtime bar before saying prefix-RFT is ready for actual experiments.
- Validation target:
  - real-data prefix run reaches first `update_actor`.
  - then runs for at least 100-200 steps without crash.
  - prefix metrics are visible and nonzero.
  - if custom prefix PPO clipping is enabled, its prefix metrics are visible too.
- References:
  - `prefix-rft-jobs/code-qwen3-0p6b-prefix-rft-v3.sh`
  - `prefix_rft_v3/ray_trainer.py:1714`
  - `prefix_rft_v3/dp_actor.py:327`
- Acceptance:
  - `actor/num_prefix_tokens > 0`
  - `actor/off_ratio > 0`
  - `actor/pg_clipfrac_prefix` is visible when `prefix_v2` loss is enabled
  - training does not regress baseline behavior

### P2-9 Clarify Dataset Feedback Loop Scope

- Goal:
  - remove ambiguity about whether `avg_rewards`, `prefix_ratios`, and `training_metrics` are active control signals or only reserved state.
- Current status:
  - the dataset stores those fields and the trainer exposes `on_batch_end`, but the current controller implementation is primarily driven by `global_step`.
- Implementation options:
  - either implement `on_batch_end(batch)` and truly update dataset history by `uid`, or
  - explicitly keep these as reserved state and document that the current prefix schedule is step-driven.
- References:
  - `prefix_rft_v3/rl_dataset.py:141`
  - `prefix_rft_v3/rl_dataset.py:156`
  - `prefix_rft_v3/rl_dataset.py:262`
  - `prefix_rft_v3/ray_trainer.py:2034`
  - `prefix_rft_v3/scheduler/global_step.py:125`
- Acceptance:
  - there is no ambiguity about whether reward-history-adaptive prefix scheduling is actually live.

### Execution Order

1. Finish `P0-1` through `P0-3` to lock the final recipe semantics.
2. Implement `P0-4` and `P0-5` so the optional prefix-specific PPO clipping path exists behind a prefix-only config switch.
3. Finish `P1-6` and `P1-7` for observability and clean launch behavior.
4. Run `P1-8` on real data.
5. Only then decide whether `P2-9` is needed for the current experiment phase.

## Implementation Update - Shared-Advantage Semantics And Prefix PPO Extension

- Locked the shared-advantage recipe in `prefix_rft_v3/ray_trainer.py`.
  - Ordinary `GRPO(uid)` remains the active advantage path.
  - Added an explicit note in `compute_advantage(...)` that prefix-specific behavior belongs in actor-side reshape/masking rather than prefix-aware advantage estimators.
- Added explicit hybrid-trajectory semantic validation in `prefix_rft_v3/ray_trainer.py`.
  - Introduced `validate_prefix_hybrid_tensors(...)`.
  - The finalized batch path now asserts:
    - `prefix_mask` is a subset of `response_mask`
    - prefix tokens participate in response loss
    - prefix tokens have non-negative `prefix_relpos`
    - non-prefix continuation tokens carry `prefix_relpos == -1`
- Implemented the optional prefix-specific PPO clipping extension in `prefix_rft_v3/core_algos.py`.
  - Registered a new policy loss mode: `prefix_v2`.
  - The implementation is based on upstream `vanilla` PPO loss and adds only the minimal prefix-specific extensions:
    - `prefix_mask`
    - `clip_ratio_low_off`
    - `clip_ratio_high_off`
    - `enable_clip`
    - prefix-specific clip metrics
  - This extension does not change the shared-advantage recipe.
- Registered custom prefix core algos at startup.
  - `prefix_rft_v3/main.py` now imports `prefix_rft_v3.core_algos` eagerly so the custom policy loss mode is always registered before training begins.
- Wired the custom prefix policy loss into the actor path.
  - `prefix_rft_v3/dp_actor.py` now forwards `prefix_mask` into the policy loss only when `policy_loss.loss_mode=prefix_v2`.
  - Existing actor-side reshape and `replace_old_log_prob` behavior remain intact.
- Wired prefix-only loss configuration through the prefix worker path.
  - `prefix_rft_v3/fsdp_workers.py` now switches the prefix actor policy loss mode to `prefix_v2` without modifying the baseline actor path.
  - The dataclass actor config receives `prefix_rft` as a post-conversion attached attribute rather than injecting unsupported top-level config keys.
- Added prefix-only config defaults in `prefix_rft_v3/config/prefix_rft_trainer.yaml`.
  - Added:
    - `policy_loss_mode: prefix_v2`
    - `clip_ratio_low_off: null`
    - `clip_ratio_high_off: null`
    - `enable_clip: True`
- Updated the prefix launch script in `prefix-rft-jobs/code-qwen3-0p6b-prefix-rft-v3.sh`.
  - The prefix job now explicitly enables:
    - `prefix_rft.policy_loss_mode=prefix_v2`
    - `prefix_rft.enable_clip=true`
  - Baseline launch behavior remains unchanged.

### Validation Completed For This Pass

- Syntax validation succeeded for:
  - `prefix_rft_v3/core_algos.py`
  - `prefix_rft_v3/main.py`
  - `prefix_rft_v3/dp_actor.py`
  - `prefix_rft_v3/fsdp_workers.py`
  - `prefix_rft_v3/ray_trainer.py`
- Registry and invocation validation succeeded for the new prefix policy loss mode.
  - Confirmed `get_policy_loss_fn("prefix_v2")` resolves.
  - Confirmed metrics include:
    - `actor/pg_clipfrac_prefix`
    - `actor/pg_clipfrac_lower_prefix`
- Hybrid-trajectory semantic assertion validation succeeded on a synthetic tensor example.

### Remaining Immediate Runtime Check

- The next runtime validation should be a real-data prefix smoke run to confirm:
  - the actor path reaches `update_actor`
  - `prefix_v2` policy loss executes inside the real prefix worker path
  - `actor/pg_clipfrac_prefix` appears in training logs
