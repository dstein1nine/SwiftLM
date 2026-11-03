// StreamTuning.swift — per-model expert-streaming settings measured on this machine.
//
// The best slot count and prefetch shape differ by model (K2.6 routes 8 of 384 experts,
// DeepSeek V4.1 6 of 384, and their experts differ in size). `scripts/autotune.py` tries
// settings on a model and saves the fastest to ~/.swiftlm/stream-tuning/<key>.json:
//
//     {"env": {"MLX_MOE_CACHE_SLOTS": "16", "MLX_MOE_PREFETCH_TOPK": "6"}, ...}
//
// With --stream-experts the server applies that file before loading. A variable already
// set in the environment wins, so a manual override still works.

import Foundation

enum StreamTuning {
    /// The streaming variables a tuning file may set; anything else in it is ignored.
    static let tunableKeys: Set<String> = [
        "MLX_MOE_CACHE_SLOTS", "MLX_MOE_PREFETCH_TOPK", "MLX_MOE_PREFETCH_DEPTH",
        "MLX_MOE_PREFETCH_MIN_LAYER",
    ]

    static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".swiftlm/stream-tuning")
    }

    /// The tuning file for a model directory: its real path with "/" as "_", so a
    /// symlinked path (~/stream) finds the same file. `scripts/autotune.py` matches this.
    static func file(for modelDirectory: URL, in directory: URL = defaultDirectory) -> URL {
        let path = modelDirectory.resolvingSymlinksInPath().standardizedFileURL.path
        let key = path.split(separator: "/").joined(separator: "_")
        return directory.appendingPathComponent("\(key).json")
    }

    /// The tuned variables for `modelDirectory` that `environment` doesn't already set.
    /// Empty when there is no tuning file or it can't be read.
    static func settings(
        for modelDirectory: URL, environment: [String: String],
        in directory: URL = defaultDirectory
    ) -> [String: String] {
        let url = file(for: modelDirectory, in: directory)
        guard let data = try? Data(contentsOf: url) else { return [:] }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let env = json["env"] as? [String: Any]
        else {
            print("[SwiftLM] ⚠️  Ignoring unreadable stream tuning file \(url.path)")
            return [:]
        }
        var result: [String: String] = [:]
        for (key, value) in env where tunableKeys.contains(key) && environment[key] == nil {
            switch value {
            case let s as String: result[key] = s
            case let n as NSNumber: result[key] = n.stringValue
            default: continue
            }
        }
        return result
    }

    /// Sets the tuned variables for `modelDirectory`. Must run before the model loads,
    /// because the streaming layers read these once.
    static func apply(modelDirectory: URL) {
        let tuned = settings(for: modelDirectory, environment: ProcessInfo.processInfo.environment)
        guard !tuned.isEmpty else { return }
        for (key, value) in tuned { setenv(key, value, 0) }
        let summary = tuned.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }
        print("[SwiftLM] Stream tuning: \(summary.joined(separator: " ")) (from \(file(for: modelDirectory).path))")
    }
}
