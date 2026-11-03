import XCTest
import MLX
import MLXLMCommon
@testable import SwiftLM

/// Hybrid (MambaCache + KVCacheSimple) models such as Qwen3.5/3.6 got no prompt reuse,
/// so every agent turn re-prefilled the whole conversation. Recurrent state can't be
/// trimmed, so reuse is exact-prefix only, snapshotted at a turn boundary.
final class HybridPromptCacheTests: XCTestCase {

    private let imStart = 7

    private func makeHybridCache(seqLen T: Int, fill: Float) -> [any KVCache] {
        let attn = KVCacheSimple()
        _ = attn.update(keys: MLXArray.ones([1, 2, T, 4], dtype: .float16) * fill,
                        values: MLXArray.ones([1, 2, T, 4], dtype: .float16) * fill)
        let mamba = MambaCache()
        mamba.state = [MLXArray.ones([1, 3, 8]) * fill, MLXArray.ones([1, 2, 4, 4]) * fill]
        return [attn, mamba]
    }

    // MARK: - Boundary

    func testBoundaryIsLastImStartForHybridCache() {
        let tokens = [imStart, 1, 2, imStart, 3, 4, imStart, 5]
        XCTAssertEqual(hybridCacheBoundary(promptTokens: tokens, imStartId: imStart,
                                           cache: [KVCacheSimple(), MambaCache()]), 6)
    }

    func testNoBoundaryForNonHybridOrNonChatML() {
        let tokens = [imStart, 1, imStart, 2]
        XCTAssertNil(hybridCacheBoundary(promptTokens: tokens, imStartId: imStart, cache: [KVCacheSimple()]),
                     "pure-attention models keep the generic prompt cache")
        XCTAssertNil(hybridCacheBoundary(promptTokens: tokens, imStartId: nil,
                                         cache: [KVCacheSimple(), MambaCache()]))
        XCTAssertNil(hybridCacheBoundary(promptTokens: [imStart, 1], imStartId: imStart,
                                         cache: [KVCacheSimple(), MambaCache()]),
                     "no history before the only turn start")
    }

    // MARK: - Save / exact-prefix restore

    func testRestoresExactPrefixIntoFreshCache() async {
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3], cache: makeHybridCache(seqLen: 3, fill: 2), allowRecurrent: true)

        let fresh: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 4, 5], limit: 4, into: fresh)

        XCTAssertEqual(n, 3)
        XCTAssertEqual(fresh[0].offset, 3, "attention layer resumes at the snapshot length")
        let recurrent = fresh[1].state
        XCTAssertEqual(recurrent.count, 2)
        XCTAssertEqual(recurrent[1].sum().item(Float.self), 2 * 32, "recurrent state restored verbatim")
    }

    func testMissesWhenPrefixDiverges() async {
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3], cache: makeHybridCache(seqLen: 3, fill: 1), allowRecurrent: true)
        let n = await pc.restoreExactPrefix(newTokens: [1, 9, 3, 4], limit: 4,
                                            into: [KVCacheSimple(), MambaCache()])
        XCTAssertNil(n, "recurrent state cannot be rolled back to a shorter common prefix")
    }

    func testMissesWhenSnapshotExtendsPastLimit() async {
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3], cache: makeHybridCache(seqLen: 3, fill: 1), allowRecurrent: true)
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 4], limit: 2,
                                            into: [KVCacheSimple(), MambaCache()])
        XCTAssertNil(n)
    }

    func testGenericSaveStillRefusesRecurrentState() async {
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3], cache: makeHybridCache(seqLen: 3, fill: 1))
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 4], limit: 4,
                                            into: [KVCacheSimple(), MambaCache()])
        XCTAssertNil(n, "only the boundary snapshot may persist recurrent state")
    }

    // MARK: - --ctx-size: RotatingKVCache attention layers

    /// A ring fed `n` single tokens; values encode the token index.
    private func makeRing(maxSize: Int, tokens n: Int) -> RotatingKVCache {
        let ring = RotatingKVCache(maxSize: maxSize, keep: 0, step: 4)
        for t in 0 ..< n {
            let k = MLXArray([Float(t)]).reshaped([1, 1, 1, 1])
            _ = ring.update(keys: k, values: k)
        }
        return ring
    }

    private func makeMamba() -> MambaCache {
        let mamba = MambaCache()
        mamba.state = [MLXArray.ones([1, 3, 8]), MLXArray.ones([1, 2, 4, 4])]
        return mamba
    }

    func testBoundaryAcceptsRotatingAttentionLayers() {
        let tokens = [imStart, 1, imStart, 2]
        XCTAssertEqual(hybridCacheBoundary(promptTokens: tokens, imStartId: imStart,
                                           cache: [RotatingKVCache(maxSize: 16), MambaCache()]), 2,
                       "--ctx-size must not turn the hybrid cache off")
    }

    func testRestoresWrappedRingExactly() async {
        let ring = makeRing(maxSize: 16, tokens: 40)  // wrapped
        let pc = PromptCache()
        await pc.save(tokens: Array(0 ..< 40), cache: [ring, makeMamba()], allowRecurrent: true)

        // Decode on the live ring after saving; the snapshot must not see it.
        for t in 40 ..< 50 {
            let k = MLXArray([Float(t)]).reshaped([1, 1, 1, 1])
            _ = ring.update(keys: k, values: k)
        }
        let fresh: [any KVCache] = [RotatingKVCache(maxSize: 16), MambaCache()]
        let n = await pc.restoreExactPrefix(newTokens: Array(0 ..< 40) + [999], limit: 40, into: fresh)

        XCTAssertEqual(n, 40)
        XCTAssertEqual(fresh[0].offset, 40)
        XCTAssertEqual(fresh[0].state[0].asArray(Float.self).sorted(), (24 ..< 40).map { Float($0) },
                       "the restored window is the snapshot's, not the live ring's")
    }

    func testRotatingHybridMissesWhenPrefixDiverges() async {
        let pc = PromptCache()
        await pc.save(tokens: Array(0 ..< 8), cache: [makeRing(maxSize: 16, tokens: 8), makeMamba()],
                      allowRecurrent: true)
        let n = await pc.restoreExactPrefix(newTokens: [0, 1, 9, 3, 4, 5, 6, 7, 8], limit: 9,
                                            into: [RotatingKVCache(maxSize: 16), MambaCache()])
        XCTAssertNil(n)
    }

    /// A ring restored from a wrapped snapshot must keep behaving exactly like the
    /// uninterrupted live ring: same offset, metaState and window after further prefill
    /// chunks and decode steps (with and without a `keep` prefix).
    func testRestoredWrappedRingContinuesLikeLiveRing() async {
        for keep in [0, 2] {
            let live = RotatingKVCache(maxSize: 16, keep: keep, step: 4)
            for t in 0 ..< 40 {
                let k = MLXArray([Float(t)]).reshaped([1, 1, 1, 1])
                _ = live.update(keys: k, values: k)
            }
            let pc = PromptCache()
            await pc.save(tokens: Array(0 ..< 40), cache: [live, makeMamba()], allowRecurrent: true)

            let restored = RotatingKVCache(maxSize: 16, keep: keep, step: 4)
            let n = await pc.restoreExactPrefix(newTokens: Array(0 ..< 40) + [999], limit: 40,
                                                into: [restored, MambaCache()])
            XCTAssertEqual(n, 40)

            for ring in [live, restored] {
                let chunk = MLXArray((100 ..< 105).map { Float($0) }).reshaped([1, 1, 5, 1])
                _ = ring.update(keys: chunk, values: chunk)
                for t in 200 ..< 206 {
                    let k = MLXArray([Float(t)]).reshaped([1, 1, 1, 1])
                    _ = ring.update(keys: k, values: k)
                }
            }
            XCTAssertEqual(restored.offset, live.offset, "keep=\(keep)")
            XCTAssertEqual(restored.metaState, live.metaState, "keep=\(keep)")
            XCTAssertEqual(restored.state[0].asArray(Float.self), live.state[0].asArray(Float.self),
                           "keep=\(keep)")
        }
    }

    // MARK: - Snapshot-only caches (DeepSeek V4.1)

    /// Stands in for DeepseekV41LayerCache: restorable through state/metaState, never trimmable.
    private final class SnapshotOnlyCache: BaseKVCache, PrefixSnapshotKVCache {
        var rows: MLXArray?
        override var state: [MLXArray] {
            get { [rows ?? MLXArray.zeros([0])] }
            set { rows = newValue[0].size == 0 ? nil : newValue[0] }
        }
        override var metaState: [String] {
            get { [String(offset)] }
            set { offset = Int(newValue[0]) ?? 0 }
        }
    }

    private let assistant = 9

    func testSnapshotBoundaryIsLastAssistantHeader() {
        let tokens = [1, assistant, 2, 3, assistant, 4]
        XCTAssertEqual(snapshotCacheBoundary(promptTokens: tokens, assistantId: assistant,
                                             cache: [SnapshotOnlyCache(), SnapshotOnlyCache()]), 4)
        XCTAssertNil(snapshotCacheBoundary(promptTokens: tokens, assistantId: assistant,
                                           cache: [SnapshotOnlyCache(), KVCacheSimple()]),
                     "every layer must be snapshot-only")
        XCTAssertNil(snapshotCacheBoundary(promptTokens: tokens, assistantId: nil,
                                           cache: [SnapshotOnlyCache()]))
        XCTAssertNil(snapshotCacheBoundary(promptTokens: [assistant, 1], assistantId: assistant,
                                           cache: [SnapshotOnlyCache()]),
                     "no history before the only assistant header")
    }

    func testSnapshotOnlyCacheRestoresExactPrefixOnly() async {
        let pc = PromptCache()
        let live = SnapshotOnlyCache()
        live.rows = MLXArray.ones([1, 3, 4]) * 5
        live.offset = 3
        await pc.save(tokens: [1, 2, 3], cache: [live], allowRecurrent: true)

        let fresh = SnapshotOnlyCache()
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 4], limit: 3, into: [fresh])
        XCTAssertEqual(n, 3)
        XCTAssertEqual(fresh.offset, 3)
        XCTAssertEqual(fresh.rows?.sum().item(Float.self), 60)

        let diverged = await pc.restoreExactPrefix(
            newTokens: [1, 8, 3, 4], limit: 3, into: [SnapshotOnlyCache()])
        XCTAssertNil(diverged, "a snapshot cannot be rewound to a shorter prefix")
    }
}
