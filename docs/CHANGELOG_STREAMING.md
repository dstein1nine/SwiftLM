# Streaming changelog

## Phase 1 — stream engine (2026-10-27)

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
- Benchmark (quiet machine, warm page cache, 3 runs × 3 prompts × 256 tokens,
  `.build/bench/phase1-final`, `CONFIGS="stream stream_stacked
  stream_stacked_sync" scripts/profiling/phase0_baseline.sh`):

  | config | decode tok/s (median [min–max]) | RSS peak | expert hit rate |
  |---|---|---|---|
  | `stream` (legacy path, unchanged, control) | 18.26 [18.21–18.44] | 4.3 GB | – |
  | `stream_stacked` | 25.65 [25.17–26.58] | 4.8 GB | 51% |
  | `stream_stacked_sync` (`MLX_MOE_MISS_SYNC=1`) | 24.27 [24.21–25.20] | 4.8 GB | 51% |

  All three produce byte-identical output to Phase 0 on all prompts. The
  control ran 14% faster than in Phase 0 (16.06), so Phase 0 was itself
  partly skewed by other load; compare within a run. Stacked is now 40%
  faster than the legacy path (24% in Phase 0); the per-miss sync costs 5%,
  so it stays off.
- The stacked path is now the default whenever `--stream-experts` is on.
  `MLX_MOE_STACKED=0` restores the legacy per-expert path. No CLI flag
  changed. Rerun (`.build/bench/phase1-default`, `CONFIGS="stream
  stream_legacy"`): `stream` 25.62 [25.26–26.22] tok/s, 4.8 GB RSS;
  `stream_legacy` 18.12 [18.05–18.38], 4.3 GB. Both byte-identical to
  Phase 0.
- `phase0_baseline.sh` now runs `swift build -c release` first. Before, it
  benchmarked whatever binary was already built, which hid the default
  change on the first rerun.
- Tests: mlx-swift-lm `MLXLMTests` 884 executed, 0 failures (8 skipped,
  serial mode); SwiftLM `make test` 326 executed, 0 failures.

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
