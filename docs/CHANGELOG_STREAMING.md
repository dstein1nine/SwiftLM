# Streaming changelog

## Phase 2 — large MoE models (2026-10-28 to 2026-11-05)

Three models that don't fit in RAM, on an M3 Ultra with 256 GB and the
internal SSD (~6.6–6.9 GB/s):

| Model | Size | Decode | 4K-token prompt | Decode at start |
|---|---|---|---|---|
| DeepSeek V4.1 Flash, MLX 4-bit | 453 GB | 5.0–6.0 tok/s | ~33 s | 4.7 tok/s |
| Kimi K2.6 DQ3 | 438 GB | 2.9–3.0 tok/s | ~56 s | 0.5–1.5 tok/s |
| GLM-5.3, MLX 4-bit | 418 GB | 2.2–2.4 tok/s | ~51 s | 1.4 tok/s |

Decode is greedy with 120–300-token replies; prompts are fresh (no prompt
cache). Branch `upstream-merge` in all three repos (SwiftLM, mlx-swift-lm,
mlx-swift), which also carries upstream SharpAI/SwiftLM `main` as of
2026-11-02.

### Model support

- GLM-5.3 (`glm_moe_dsa`) and Kimi K2.6 (`kimi_k2`) load through the
  DeepseekV32 model. GLM needed its absorbed MLA attention ported; Kimi needed
  `language_model.model.`-prefixed expert names resolved for streaming.
- DeepSeek V4.1 Flash runs on the upstream overlay, plus: Engram memory
  (hashed tables read from disk, checked against a NumPy port of the
  reference), MLX checkpoints' 8-bit attention, and streaming of per-expert
  (unstacked) checkpoints.
- Chat templates: Kimi takes the thinking switch as `thinking`, and
  `reasoning_effort` is passed to templates. GLM's template has no off switch
  for thinking; that's by design.

### Decode

- **Wired memory** (SwiftLM 387aed9). SwiftLM never set an MLX wired limit,
  so the kernel re-established GPU residency on every submission. Kimi
  decode: 0.53 → 2.65 tok/s. With `--stream-experts` the server now wires
  Metal's recommended working set (`--wired-limit-gb`, 0 disables). Any
  standalone harness needs the same `WiredMemoryTicket`, or it runs up to 4×
  slower.
- **Hits before misses** (mlx-swift-lm 5593fe6). Slot resolution assigned
  miss slots before it checked for hits, and evicted experts that were about
  to be used. Kimi 2.68 → 2.96 tok/s.
- **Next-layer prefetch** (73eb8b6, be10a6b, 001e7bf). Runs layer i+1's
  router on layer i's hidden state and reads its predicted experts early. On
  by default: top 6, from layer 7, 1 layer ahead. Kimi 2.87 → 3.00, GLM
  2.12 → 2.24 tok/s. Small, because one layer's GPU work hides only ~1.4 ms
  of reads.
- **Fewer command buffers** (56942cc, SwiftLM 9f0f271). MLX commits a command
  buffer per 50 MB of inputs, and stacked expert buffers exceed that. The
  server raises `MLX_MAX_MB_PER_BUFFER` when streaming. Encode CPU −25%,
  decode +1%.
- **Per-expert reads by file offset** (c8db812, cb81ba0). Per-expert
  checkpoints silently fell back to the legacy path, and their first read of
  each tensor re-parsed the shard header under a global lock. V4.1 decode
  4.7 → 6.3 tok/s; first 4K-token prefill after a start 196 → 50 s.
- **quantized_matmul casts on the op's stream** (mlx-swift 3f1fede). The
  dtype casts inside `quantized_matmul` ran on the thread's default stream
  instead of the matmul's, so every expert matmul crossed streams: ~1,400
  fence waits per GLM token. Output byte-identical. V4.1 +13–14%, GLM +3.5%,
  Kimi +2–3%.
- **Per-model tuning file** (SwiftLM 0d4ee33, 04e40a5).
  `~/.swiftlm/stream-tuning/<model path>.json` sets the `MLX_MOE_*` knobs and
  `prefill_size` at startup; environment variables win. Autotuning V4.1 and
  GLM kept the defaults (16 slots, top 6, depth 1, from layer 7).

### Prefill

- **Bigger chunks** (SwiftLM af685ac, tuning `prefill_size`). Each prompt
  chunk reads every expert it routes to, and a 512-token chunk already
  routes to nearly all of them. So each chunk cost one full pass over the
  experts (~360–420 GB). Kimi 2.9K tokens: 305 s at 512 → 91 s at 4096.
  Default 4096 with `--stream-experts`; the tuning files set 16384.
- **Prompt pipeline** (6678932, b0314e1). While a layer computes, the
  next layer's predicted experts for the chunk are read in the background
  (two alternating buffer sets). Only on chunks of 512+ tokens; below that,
  wrong guesses cost more than the overlap saves. 4K tokens: V4.1 46 → 33 s,
  Kimi 80 → 56 s, GLM 81 → 51 s.
- **Prompt cache for DSA and V4.1** (SwiftLM cc86df7, 80c610d, f6de05d).
  Restore treated DSA `CacheList` layers as recurrent and always missed, so
  every chat turn prefilled the whole conversation. Restore now works for
  those layers and for V4.1, and the reply is saved with the prompt.
  Follow-up turns: Kimi 15–29 s (growing each turn) → 7–10 s, V4.1 ~90 s →
  7–9 s.

### Correctness

- **DeepseekV3 gate** (mlx-swift-lm 8dc7533). Expert weights came from the
  bias-corrected scores, which should only choose the experts. GLM's biases
  are 8–34 against scores below 1, so its 8 experts were weighted almost
  equally. Teacher-forced perplexity: GLM prose 3.46 → 2.91, code 4.82 →
  4.52; Kimi (biases ~0.06) within noise. `DeepseekV3RoutingTests` fails on
  the old gate. V4.1 has its own router and wasn't affected.
- Stacked slot pool: reuse checked only the gate shape, which crashed Kimi's
  mixed-bit `down_proj` layers, and decode never released the prompt pool
  (Kimi decode −6%). Both fixed in b0314e1.
- MLX registers the default stream globally, not per creating thread (mlx-swift
  5da66ad).

### Tried and dropped

Each was measured and gave no gain:

- **More in-process cache slots.** Each extra slot comes out of the OS page
  cache, which is the better cache: 16 slots 3.08 tok/s, 32 2.90, 48 2.65
  (Kimi).
- **Smarter cache policies.** On a replayed V4.1 decode trace, the page
  cache's LRU is near the best possible. Even Belady-optimal only halves
  misses, and LFU is worse because routing drifts with the topic.
- **Packing an expert's tensors together.** Three scattered 5.9 MB reads run
  as fast as one 17.7 MB read; the SSD is saturated either way.
- **Zero-copy Metal buffers over mmapped experts.** A page-cache hit costs
  the same as the pread copy, and uncached it's 2× slower.
- **Prefetch started earlier, or deeper than 1 layer.** More read
  contention, same misses.
- **GPU keep-alive** against clock-down: +1.7% on GLM.
- **MTP speculative decoding** (GLM-5.3's MTP layer, from the BF16
  checkpoint). 88–94% accepted, but a 2-token verify step routes to ~2× the
  experts, so it costs ~1.9× a 1-token step. 2.15 → 2.17 tok/s.
- **Prompt reads that skip the page cache** (`F_NOCACHE`; resident pages
  are still used). Prompts under 1K tokens got ~3 s faster, but decode lost
  the pages prefill used to leave behind (2.34 → 2.27 tok/s), so a turn
  breaks even.

### Where the limits are

- **Decode is bound by expert reads.** On GLM the GPU is busy ~21% of the
  time. Of the idle time, ~75% is waiting on expert reads, ~21% graph build
  and encode, ~5% GPU sync. The disk averages 4.2 GB/s: full speed while a
  layer's misses load, idle otherwise, because a layer can't read before its
  router runs. Perfect overlap would cap GLM at ~3.5 tok/s, and prefetch
  already captures what a 1-layer lookahead can.
- **Prefill is bound by disk bandwidth.** Past ~100 tokens the disk runs at
  5.5–6.3 GB/s, and a chunk reads each expert it needs once. Even a
  153-token GLM prompt touches most of the 256 experts per layer and reads
  130–160 GB. The page cache serves only ~16% of prompt reads.
- The remaining levers are fewer bytes per expert (a smaller quant), more
  RAM for the page cache, or a faster SSD.

### Benchmarking notes

- Check disk0 read bytes (iostat) on every run. The OS page cache serves
  much of what the streamer reads, so warm runs measure overhead, not the
  SSD. Before comparing page-cache-sensitive configs, flush by reading
  another model's files.
- Discard the first request after a server start.
- Time Profiler samples only on-CPU threads, so time blocked on reads
  doesn't show up as samples; infer it from the gaps.
- GLM output can't be token-identical across prefill, decode and multi-token
  steps. bf16 rounding flips near-tie expert choices, and the difference
  compounds over 78 layers. Compare hashes within one code path.

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
