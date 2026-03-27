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
