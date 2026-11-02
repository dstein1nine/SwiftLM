// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
//
// Registers SharpAI overlay text models that mlx-swift-lm ships but does not wire into
// LLMTypeRegistry (their creators live in MLXVLM, which MLXLLM cannot import).
//
// Called once at startup, before any model loading.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM

/// Register overlay model types served as text-only LLMs.
///
/// deepseek_v41 (DeepSeek V4.1 Flash): the creator still rejects checkpoints with Engram
/// layers, which every published V4.1 Flash checkpoint has, so loading one fails with that
/// error until Engram is implemented.
func registerOverlayModelTypes() async {
    let registry = LLMTypeRegistry.shared
    if let create = SharpAIDeepseekV41Overlay.creators["deepseek_v41"] {
        await registry.registerModelType("deepseek_v41") { data in try create(data) }
    }
}
