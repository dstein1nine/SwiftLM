import XCTest
@testable import SwiftLM

/// Kimi K2.x MLX repos ship only `tiktoken.model`; the converted `tokenizer.json` under
/// models/_tokenizers/kimi-k2 (see convert_kimi_tokenizer.py there) must load through
/// the server's real tokenizer loader and reproduce tiktoken's ids exactly, including
/// the reserved special tokens and the `&&`-intersection split regex.
final class KimiTokenizerTests: XCTestCase {

    func testConvertedKimiTokenizerMatchesTiktoken() async throws {
        let folder = URL(
            fileURLWithPath: NSString(string: "~/dev/stream/models/_tokenizers/kimi-k2")
                .expandingTildeInPath)
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: folder.appendingPathComponent("tokenizer.json").path),
            "converted Kimi tokenizer not present")

        let tokenizer = try await TransformersTokenizerLoader(modelId: "kimi-k2").load(from: folder)
        let expected: [(String, [Int])] = [
            ("Hello world", [19180, 2695]),
            ("I can't believe it's 2026!", [40, 8971, 6613, 4643, 220, 2975, 21, 0]),
            ("你好，世界！这是测试。", [33845, 378, 2243, 856, 2953, 9555, 292]),
            ("def f(x):\n    return x**2\n", [1166, 285, 3940, 2360, 274, 745, 1288, 381, 17, 198]),
            (
                "<|im_user|>user<|im_middle|>What is 2+2?<|im_end|>",
                [163587, 2482, 163601, 5376, 387, 220, 17, 10, 17, 30, 163586]
            ),
            (
                "Solve ∫₀¹ x² dx = 1/3. Prove √2 is irrational.",
                [
                    91351, 86701, 20535, 33370, 1288, 39530, 26669, 327, 220, 16, 14, 18, 13,
                    18446, 60329, 17, 387, 113607, 13,
                ]
            ),
        ]
        for (text, ids) in expected {
            let got = tokenizer.encode(text: text, addSpecialTokens: false)
            XCTAssertEqual(got, ids, text)
            XCTAssertEqual(tokenizer.decode(tokenIds: got, skipSpecialTokens: false), text, text)
        }
        XCTAssertEqual(tokenizer.convertTokenToId("[EOS]"), 163585)
        XCTAssertEqual(tokenizer.convertTokenToId("<|im_end|>"), 163586)
    }
}
