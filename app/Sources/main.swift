import AppKit
import Foundation

// Entry point.
//   Farabi --watchdog                      windowless background watchdog (started by launchd)
//   Farabi --test-chat <model> <question>  headless test: one request to the engine, prints reply and stats
//   (no arguments)                         the normal windowed app
let arguments = CommandLine.arguments
if arguments.contains("--watchdog") {
    Watchdog.run()
} else if let i = arguments.firstIndex(of: "--test-chat"), arguments.count > i + 2 {
    let model = arguments[i + 1], question = arguments[i + 2]
    let finished = DispatchSemaphore(value: 0)
    Task.detached {
        var text = ""
        do {
            for try await chunk in Engine.chat(model: model, messages: [["role": "user", "content": question]], think: false) {
                text += chunk.content
                if let e = chunk.error { print("ERROR:", e) }
                if chunk.done { print("REPLY:", text); print("STATS:", chunk.stats ?? "") }
            }
        } catch { print("ERROR:", error.localizedDescription) }
        finished.signal()
    }
    finished.wait()
    exit(0)
} else {
    FarabiApp.main()
}
