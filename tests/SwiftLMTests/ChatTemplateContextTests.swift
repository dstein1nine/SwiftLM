import XCTest
import MLXLMCommon
@testable import SwiftLM

/// The server forwards thinking/reasoning switches to the chat template through
/// `additionalContext`. Each model family reads a different variable, so these tests
/// render the real templates shipped in the local model folders and check that the
/// variable the server passes actually changes the prompt.
///
/// - Kimi K2.x: `thinking == false` must emit an empty `<think></think>` block, and the
///   tool branch must render (the stock template used `tojson(separators=...)`, which the
///   Swift Jinja engine rejects; the local copy drops the argument).
/// - GLM-5.x: no thinking off-switch; `reasoning_effort` drives a "Reasoning Effort:" line.
final class ChatTemplateContextTests: XCTestCase {

    private func loadTokenizer(_ relativePath: String) async throws -> any MLXLMCommon.Tokenizer {
        let folder = URL(
            fileURLWithPath: NSString(string: "~/dev/stream/models/\(relativePath)")
                .expandingTildeInPath)
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: folder.appendingPathComponent("tokenizer.json").path),
            "\(relativePath) tokenizer not present")
        return try await TransformersTokenizerLoader(modelId: relativePath).load(from: folder)
    }

    private let messages: [[String: any Sendable]] = [
        ["role": "user", "content": "What is 2+2?"]
    ]

    func testKimiThinkingFalseEmitsEmptyThinkBlock() async throws {
        let tok = try await loadTokenizer("Kimi-K2.6-DQ3")
        let off = try tok.applyChatTemplate(
            messages: messages, tools: nil,
            additionalContext: ["enable_thinking": false, "thinking": false])
        let offText = tok.decode(tokenIds: off, skipSpecialTokens: false)
        XCTAssertTrue(offText.hasSuffix("<think></think>"), offText)

        let on = try tok.applyChatTemplate(
            messages: messages, tools: nil,
            additionalContext: ["enable_thinking": true, "thinking": true])
        let onText = tok.decode(tokenIds: on, skipSpecialTokens: false)
        XCTAssertFalse(onText.hasSuffix("<think></think>"), onText)
    }

    func testKimiToolBranchRenders() async throws {
        let tok = try await loadTokenizer("Kimi-K2.6-DQ3")
        let tools: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": [
                        "type": "object",
                        "properties": ["city": ["type": "string"]],
                        "required": ["city"],
                    ],
                ],
            ]
        ]
        let ids = try tok.applyChatTemplate(
            messages: messages, tools: tools,
            additionalContext: ["enable_thinking": false, "thinking": false])
        let text = tok.decode(tokenIds: ids, skipSpecialTokens: false)
        XCTAssertTrue(text.contains("tool_declare"), text)
        XCTAssertTrue(text.contains("get_weather"), text)
    }

    func testGLMReasoningEffortReachesTemplate() async throws {
        let tok = try await loadTokenizer("GLM-5.3-4bit")
        let low = try tok.applyChatTemplate(
            messages: messages, tools: nil,
            additionalContext: ["enable_thinking": false, "thinking": false, "reasoning_effort": "low"])
        let lowText = tok.decode(tokenIds: low, skipSpecialTokens: false)
        XCTAssertTrue(lowText.contains("Reasoning Effort: Low"), lowText)

        let unset = try tok.applyChatTemplate(
            messages: messages, tools: nil,
            additionalContext: ["enable_thinking": false, "thinking": false])
        let unsetText = tok.decode(tokenIds: unset, skipSpecialTokens: false)
        XCTAssertTrue(unsetText.contains("Reasoning Effort: Max"), unsetText)
    }
}
