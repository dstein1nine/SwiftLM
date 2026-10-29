// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
//
// Registers SwiftLM-owned DFlash model types with the shared LLMTypeRegistry,
// overriding any MLXLLM defaults so DFlashTargetModel conformance is available.
//
// Called once at startup, before any model loading.

import Foundation
import MLXLLM
import MLXLMCommon

/// Register SwiftLM-owned model types that conform to DFlashTargetModel.
///
/// Must be called before any `LLMModelFactory.shared.loadContainer()` call so
/// that the factory produces SwiftLM types (which carry DFlash conformance)
/// rather than the MLXLLM defaults.
func registerDFlashModelTypes() async {
    let registry = LLMTypeRegistry.shared

    // DeepSeek V3 — override MLXLLM default with DFlash-capable version.
    await registry.registerModelType("deepseek_v3") { data in
        let config = try JSONDecoder.json5().decode(DSV3Config.self, from: data)
        return DeepseekV3DFlashModel(config)
    }

    // kimi_k2 / kimi_k25 are deliberately NOT overridden: MLXLLM maps them to the
    // DeepSeek V3.2 model, which handles the absorbed `embed_q`/`unembed_out`
    // attention layout and nested `text_config` that current Kimi K2.x MLX
    // checkpoints ship, and which conforms to StreamableMoE for --stream-experts.
    // DSV3Config / DeepseekV3DFlashModel can load neither, so DFlash is not
    // available for Kimi until the DFlash model learns that layout.

    // Kimi linear — hybrid KDA/MLA architecture (kimi 2.6).
    await registry.registerModelType("kimi_linear") { data in
        let config = try JSONDecoder.json5().decode(KimiLinearConfiguration.self, from: data)
        return KimiLinearDFlashModel(config)
    }
}
