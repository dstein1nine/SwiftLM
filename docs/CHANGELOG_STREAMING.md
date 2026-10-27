# Streaming changelog

## Phase 1 — stream engine (in progress, 2026-10-27)

- mlx-swift (add-only): `mlx_fast_pread_range_into` / `MLXFast.preadRangeInto`,
  a plain range read into an evaluated array with an optional `F_NOCACHE`
  flag. No existing function changed.
- mlx-swift-lm: new `Streaming/` module with `StreamableUnit` (an expert =
  gate/up/down extents), `SafetensorsLayout`, `ResidencyPolicy` (LRU),
  `UnitPredictor` (previous token), `ExpertSlotTable` and `StreamEngine`
  (configurable queue depth `MLX_MOE_STREAM_QD`, `MLX_MOE_STREAM_NOCACHE`,
  hit/miss/byte metrics). The stacked path (`MLX_MOE_STACKED=1`) now runs on
  the engine.
- Verified: on Qwen3.6-35B-A3B-4bit the ported stacked path, before the two
  changes below, produced byte-identical output to Phase 0 on all three
  benchmark prompts (temperature 0).
- Fix: when a decode-sized call routed to more experts than there are slots,
  the stacked path renamed slots to experts it never read, then fell back.
  Later tokens could compute with the wrong expert's weights. Slot
  resolution now rolls back on failure. A regression test
  (`testStackedSlotExhaustionLeavesSlotsUntouched`) fails without the fix.
- Removed the per-miss `print` to stdout. The per-miss
  `Stream.gpu.synchronize()` is now off by default (`MLX_MOE_MISS_SYNC=1`
  restores it). The idx readback already orders the reads after earlier uses
  of the slots.
- `/metrics` gains `swiftlm_stream_*` counters (hits, misses, prefetched,
  bytes, read seconds).
- Finding: the stacked path's previous-token prefetch never reads anything in
  steady state, because the previous token's experts are always still
  resident. All its reads are misses.
- Pending: benchmarks of the last two changes on a quiet machine (the first
  run was skewed by 186 GB of LM Studio models in memory), and making stacked
  the default.

## Phase 0 — survey and baseline (2026-10-27)

No runtime code changed. Scope narrowed to MoE models only: dense layer
streaming and the dense activation-sparsity predictor are dropped.

- Added `docs/STREAMING_ARCHITECTURE.md`: the current SSD expert-streaming
  path, file map, dead code, and where the code contradicts the project brief.
- Added `scripts/profiling/phase0_baseline.sh` and
  `scripts/profiling/streaming_bench_client.py`: decode tok/s, TTFT, peak
  RSS, MLX peak, disk0 bytes read, and output hashes per configuration.
- Added `scripts/profiling/ssd_read_bench.py`: cold random-read bandwidth
  with `F_NOCACHE`, checked against iostat.
- Baseline on M3 Ultra / 256 GB, Qwen3.6-35B-A3B-4bit, warm page cache:
  resident 79.1 tok/s, `--stream-experts` 16.1 tok/s (4.3 GB RSS), stacked
  19.9, top-k 4 20.5. SSD 6.57 GB/s at QD ≥ 6 → cold ceiling 13.7 tok/s at
  k=8.
