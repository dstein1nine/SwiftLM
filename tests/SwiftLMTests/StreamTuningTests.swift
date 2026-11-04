import Foundation
import XCTest

@testable import SwiftLM

/// The server applies `scripts/autotune.py`'s saved settings, but never over a variable
/// the user set, and only the streaming variables the tuner owns.
final class StreamTuningTests: XCTestCase {
    private var root: URL!
    private var tuningDir: URL!
    private var modelDir: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("stream-tuning-\(UUID().uuidString)")
        tuningDir = root.appendingPathComponent("tuning")
        modelDir = root.appendingPathComponent("models/Some-MoE-4bit")
        try FileManager.default.createDirectory(at: tuningDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func writeTuning(_ json: String) throws {
        try Data(json.utf8).write(to: StreamTuning.file(for: modelDir, in: tuningDir))
    }

    func testAppliesTunedKeysTheEnvironmentLeavesUnset() throws {
        try writeTuning("""
            {"env": {"MLX_MOE_CACHE_SLOTS": "24", "MLX_MOE_PREFETCH_TOPK": 8,
                     "MLX_MOE_PREFETCH_DEPTH": "2", "PATH": "/evil"},
             "decode_tok_s": 6.4}
            """)
        let settings = StreamTuning.settings(
            for: modelDir, environment: ["MLX_MOE_PREFETCH_DEPTH": "1"], in: tuningDir)
        XCTAssertEqual(settings, ["MLX_MOE_CACHE_SLOTS": "24", "MLX_MOE_PREFETCH_TOPK": "8"])
    }

    func testPrefillSizeIsReadWhenPositive() throws {
        XCTAssertNil(StreamTuning.prefillSize(for: modelDir, in: tuningDir))
        try writeTuning(#"{"env": {}, "prefill_size": 16384}"#)
        XCTAssertEqual(StreamTuning.prefillSize(for: modelDir, in: tuningDir), 16384)
        try writeTuning(#"{"env": {}, "prefill_size": 0}"#)
        XCTAssertNil(StreamTuning.prefillSize(for: modelDir, in: tuningDir))
    }

    func testNoFileOrBadFileAppliesNothing() throws {
        XCTAssertEqual(StreamTuning.settings(for: modelDir, environment: [:], in: tuningDir), [:])
        try writeTuning("not json")
        XCTAssertEqual(StreamTuning.settings(for: modelDir, environment: [:], in: tuningDir), [:])
    }

    /// ~/stream links to ~/dev/stream; both spellings must find one file.
    func testSymlinkedModelPathFindsTheSameFile() throws {
        let link = root.appendingPathComponent("linked")
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: root.appendingPathComponent("models"))
        XCTAssertEqual(
            StreamTuning.file(for: link.appendingPathComponent("Some-MoE-4bit"), in: tuningDir),
            StreamTuning.file(for: modelDir, in: tuningDir))
    }
}
