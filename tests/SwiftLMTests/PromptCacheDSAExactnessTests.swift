import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest
@testable import SwiftLM

/// The prompt cache must be exact for DSA models (Kimi K2, GLM-5, DeepSeek V3.2), whose
/// layers keep a `CacheList` of the attention KV and the sparse indexer's keys.
///
/// Continuing from a restored prefix must give the same logits, bit for bit, as continuing
/// in the cache that computed the prefix. Prefilling the whole prompt in one pass is a
/// different computation (other matmul shapes), so it is only expected to be close.
final class PromptCacheDSAExactnessTests: XCTestCase {

    /// Tiny GLM-5-shaped model. `index_topk` 4 is below the sequence length, so the
    /// sparse indexer selects keys from its cache instead of attending densely.
    private func makeModel() async throws -> any LanguageModel {
        let json = """
            {"model_type": "glm_moe_dsa", "vocab_size": 128, "hidden_size": 64,
             "intermediate_size": 128, "moe_intermediate_size": 32, "num_hidden_layers": 4,
             "num_attention_heads": 4, "num_key_value_heads": 4, "n_shared_experts": 1,
             "n_routed_experts": 4, "routed_scaling_factor": 2.5, "kv_lora_rank": 16,
             "q_lora_rank": 32, "qk_rope_head_dim": 8, "v_head_dim": 16, "qk_nope_head_dim": 16,
             "norm_topk_prob": true, "n_group": 1, "topk_group": 1, "num_experts_per_tok": 2,
             "moe_layer_freq": 1, "first_k_dense_replace": 1, "max_position_embeddings": 4096,
             "rms_norm_eps": 1e-05, "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
             "attention_bias": false, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
             "index_topk": 4, "index_n_heads": 4, "index_head_dim": 32,
             "indexer_rope_interleave": true}
            """
        return try await LLMTypeRegistry.shared.createModel(
            configuration: json.data(using: .utf8)!, modelType: "glm_moe_dsa")
    }

    func testRestoredPrefixContinuesExactly() async throws {
        MLXRandom.seed(7)
        let model = try await makeModel()
        guard let lm = model as? DeepseekV32Model else { return XCTFail("not DeepseekV32Model") }
        eval(lm)
        let prefix = (0 ..< 12).map { ($0 * 37 + 5) % 128 }
        let suffix = (0 ..< 8).map { ($0 * 53 + 11) % 128 }
        func tokens(_ t: [Int]) -> MLXArray { MLXArray(t.map(Int32.init)).reshaped(1, t.count) }

        // B: prefix then suffix in one live cache.
        let live = lm.newCache(parameters: nil)
        eval(lm(tokens(prefix), cache: live))
        let continued = lm(tokens(suffix), cache: live)

        // C: prefix, one decode token (the server saves after it), save, restore into a
        // fresh cache, then the suffix.
        let source = lm.newCache(parameters: nil)
        eval(lm(tokens(prefix), cache: source))
        eval(lm(tokens([3]), cache: source))
        let pc = PromptCache()
        await pc.save(tokens: prefix, cache: source)
        let fresh = lm.newCache(parameters: nil)
        let matched = await pc.restore(newTokens: prefix + suffix, into: fresh)
        XCTAssertEqual(matched, prefix.count)
        let restored = lm(tokens(suffix), cache: fresh)

        let diff = abs(continued - restored).max().item(Float.self)
        XCTAssertEqual(diff, 0, "restored prefix must continue exactly like the live cache")

        // A: one pass. Close, not necessarily identical.
        let onePass = lm(tokens(prefix + suffix), cache: lm.newCache(parameters: nil))
        let tail = onePass[0..., prefix.count..., 0...]
        let drift = abs(tail - continued).max().item(Float.self)
        print("[DSA cache test] restored-vs-live max diff \(diff), one-pass-vs-chunked \(drift)")
        XCTAssertLessThan(drift, 1e-2)
    }
}
