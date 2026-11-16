import Foundation

/// The Farabi engine is a Python package (`engine.server`) that serves a subset of the Ollama HTTP API on 127.0.0.1:11438.
/// This file contains everything that talks to it: status, model list, load/unload, streaming chat, starting and stopping
/// the engine process, the background watchdog (`Farabi --watchdog`) and the launchd job that runs the watchdog at login.

struct ModelInfo: Identifiable, Hashable {
    let id: String                 // the engine's model name, e.g. "qwen3.8-27b"
    let family: String
    let size: String               // parameter count, e.g. "27.8B"
    let quantization: String       // e.g. "nvfp4"
    let diskBytes: Int64
    var memoryBytes: Int64 = 0     // bytes occupied while loaded, 0 when not in memory
    var isLoaded: Bool { memoryBytes > 0 }
}

struct EngineStatus: Equatable {
    var running = false
    var version = ""
    var networkLock = false
    var busy = false
}

/// One decoded line of the streaming chat response.
struct Chunk: Sendable {
    var content = ""
    var thinking = ""
    var done = false
    var stats: String?
    var error: String?
}

enum Engine {
    /// Port of the engine. Override with the `enginePort` defaults key (the engine itself reads `FARABI_PORT` / `--port`).
    static var port: Int { let p = UserDefaults.standard.integer(forKey: "enginePort"); return p > 0 ? p : 11438 }
    static var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }

    /// Engine folder (the cloned repository root). Override with the `engineFolder` defaults key.
    static var folder: String { UserDefaults.standard.string(forKey: "engineFolder") ?? NSHomeDirectory() + "/farabi" }
    /// Python interpreter used to start the engine. Override with the `enginePython` defaults key.
    static var python: String { UserDefaults.standard.string(forKey: "enginePython") ?? NSHomeDirectory() + "/farabi/.venv/bin/python" }
    static var processLog: String { NSHomeDirectory() + "/Library/Logs/farabi-engine-process.log" }
    static var requestLog: String { NSHomeDirectory() + "/Library/Logs/farabi-engine.log" }

    /// True when the engine package and the Python interpreter exist.
    static var isInstalled: Bool {
        var isDirectory: ObjCBool = false
        let hasPackage = FileManager.default.fileExists(atPath: folder + "/engine", isDirectory: &isDirectory) && isDirectory.boolValue
        return hasPackage && FileManager.default.isExecutableFile(atPath: python)
    }

    // MARK: HTTP

    /// GET (or POST when `body` is given) against the engine; returns the decoded JSON object, or nil on any failure.
    static func json(_ path: String, body: [String: Any]? = nil, timeout: TimeInterval = 5) async -> [String: Any]? {
        var request = URLRequest(url: path.isEmpty ? baseURL : baseURL.appendingPathComponent(path), timeoutInterval: timeout)
        if let body {
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        }
        guard let (data, response) = try? await URLSession.shared.data(for: request), (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    static func status() async -> EngineStatus {
        guard let j = await json("", timeout: 1.5) else { return EngineStatus() }
        return EngineStatus(running: true, version: (j["version"] as? String) ?? "", networkLock: (j["network_lock"] as? Bool) ?? false,
                            busy: (j["busy"] as? Bool) ?? false)
    }

    static func models() async -> [ModelInfo] {
        guard let tags = await json("api/tags"), let list = tags["models"] as? [[String: Any]] else { return [] }
        var memory: [String: Int64] = [:]
        if let ps = await json("api/ps"), let loaded = ps["models"] as? [[String: Any]] {
            for m in loaded { if let name = m["name"] as? String { memory[name] = (m["size_vram"] as? NSNumber)?.int64Value ?? 1 } }
        }
        return list.compactMap { m in
            guard let name = m["name"] as? String else { return nil }
            let d = (m["details"] as? [String: Any]) ?? [:]
            return ModelInfo(id: name, family: (d["family"] as? String) ?? "", size: (d["parameter_size"] as? String) ?? "",
                             quantization: (d["quantization_level"] as? String) ?? "", diskBytes: (m["size"] as? NSNumber)?.int64Value ?? 0,
                             memoryBytes: memory[name] ?? 0)
        }
    }

    static func load(_ id: String) async { _ = await json("api/generate", body: ["model": id, "keep_alive": "30m"], timeout: 300) }
    static func unload(_ id: String) async { _ = await json("api/generate", body: ["model": id, "keep_alive": 0], timeout: 60) }

    /// Plain chat: only the given messages are sent. No system prompt, no tools, no options (the model's own defaults apply).
    static func chat(model: String, messages: [[String: String]], think: Bool) -> AsyncThrowingStream<Chunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = URLRequest(url: baseURL.appendingPathComponent("api/chat"), timeoutInterval: 3600)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.httpBody = try JSONSerialization.data(withJSONObject: ["model": model, "messages": messages, "stream": true, "think": think])
                    let (bytes, response) = try await URLSession.shared.bytes(for: request)
                    let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                    for try await line in bytes.lines {
                        guard let d = line.data(using: .utf8), let j = (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] else { continue }
                        var chunk = Chunk()
                        if let e = j["error"] as? String { chunk.error = e }
                        if let m = j["message"] as? [String: Any] {
                            chunk.content = (m["content"] as? String) ?? ""
                            chunk.thinking = (m["thinking"] as? String) ?? ""
                        }
                        if (j["done"] as? Bool) == true {
                            chunk.done = true
                            chunk.stats = statsText(j)
                        }
                        continuation.yield(chunk)
                    }
                    if code != 200 { continuation.yield(Chunk(done: true, error: "the engine returned HTTP \(code)")) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// One-line generation statistics from the final streamed object (Ollama-style duration fields are in nanoseconds).
    private static func statsText(_ j: [String: Any]) -> String {
        func number(_ key: String) -> Double { (j[key] as? NSNumber)?.doubleValue ?? 0 }
        let generated = number("eval_count"), generationSec = number("eval_duration") / 1e9
        let promptTokens = number("prompt_eval_count"), readSec = number("prompt_eval_duration") / 1e9, loadSec = number("load_duration") / 1e9
        func decimal(_ x: Double) -> String { let s = String(format: "%.1f", x); return Lang.isTurkish ? s.replacingOccurrences(of: ".", with: ",") : s }
        var parts: [String] = []
        if generationSec > 0 { parts.append(String(format: tx("%.0f belirteç/sn", "%.0f tokens/s"), generated / generationSec)) }
        parts.append(String(format: tx("%.0f belirteç üretildi", "%.0f tokens generated"), generated))
        parts.append(String(format: tx("istem %.0f belirteç, okuma ", "prompt %.0f tokens, read "), promptTokens) + decimal(readSec) + tx(" sn", " s"))
        if loadSec > 0.5 { parts.append(tx("model yükleme ", "model load ") + decimal(loadSec) + tx(" sn", " s")) }
        return parts.joined(separator: " · ")
    }

    // MARK: Process

    /// Starts the engine as a child process (`python -m engine.server`, working directory = engine folder).
    /// The process keeps running even if the starting process exits.
    @discardableResult
    static func startProcess() -> Process? {
        guard isInstalled else { return nil }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: python)
        p.arguments = ["-m", "engine.server", "--port", String(port)]
        p.currentDirectoryURL = URL(fileURLWithPath: folder)
        if !FileManager.default.fileExists(atPath: processLog) { FileManager.default.createFile(atPath: processLog, contents: nil) }
        if let h = FileHandle(forWritingAtPath: processLog) {
            h.seekToEndOfFile()
            p.standardOutput = h
            p.standardError = h
        }
        p.standardInput = FileHandle.nullDevice
        do { try p.run() } catch { return nil }
        return p
    }

    /// Process IDs listening on the engine port, whoever started them.
    static func listeningPIDs() -> [Int32] {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        p.arguments = ["-nP", "-iTCP:\(port)", "-sTCP:LISTEN", "-t"]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        guard (try? p.run()) != nil else { return [] }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return String(decoding: data, as: UTF8.self).split(separator: "\n").compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }
    }

    static func stopProcess() {
        for pid in listeningPIDs() { kill(pid, SIGTERM) }
    }

    /// Blocking status probe (used by the watchdog and by app termination, which cannot await).
    static func probeSync(timeout: TimeInterval = 1.5) -> (running: Bool, busy: Bool) {
        let semaphore = DispatchSemaphore(value: 0)
        var result = (false, false)
        let task = URLSession.shared.dataTask(with: URLRequest(url: baseURL, timeoutInterval: timeout)) { data, response, _ in
            if (response as? HTTPURLResponse)?.statusCode == 200 {
                let j = data.flatMap { (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any] }
                result = (true, (j?["busy"] as? Bool) ?? false)
            }
            semaphore.signal()
        }
        task.resume()
        _ = semaphore.wait(timeout: .now() + timeout + 1)
        return result
    }
}

/// Background watchdog: `Farabi --watchdog` (started by launchd). Starts the engine when it is not running and restarts it
/// when it dies. It has no window.
enum Watchdog {
    static var logPath: String { NSHomeDirectory() + "/Library/Logs/farabi-watchdog.log" }

    static func log(_ message: String) {
        let f = DateFormatter()
        f.dateFormat = "MM-dd HH:mm:ss"
        let line = "\(f.string(from: Date())) \(message)\n"
        if !FileManager.default.fileExists(atPath: logPath) { FileManager.default.createFile(atPath: logPath, contents: nil) }
        if let h = FileHandle(forWritingAtPath: logPath) {
            h.seekToEndOfFile()
            h.write(Data(line.utf8))
            try? h.close()
        }
    }

    static func run() -> Never {
        log("watchdog started (pid \(getpid()))")
        var child: Process?
        var loggedMissing = false
        while true {
            if !Engine.probeSync().running {
                if !Engine.isInstalled {
                    if !loggedMissing { log("engine not found (check the engineFolder / enginePython settings) - waiting"); loggedMissing = true }
                    sleep(30)
                    continue
                }
                loggedMissing = false
                if child == nil || child?.isRunning == false {
                    child = Engine.startProcess()
                    log(child == nil ? "engine could NOT be started" : "engine started (pid \(child!.processIdentifier))")
                }
                for _ in 0..<40 {                       // importing Python and the inference stack takes a few seconds
                    usleep(500_000)
                    if Engine.probeSync().running { log("engine is up"); break }
                    if child?.isRunning == false { log("engine process exited (status \(child!.terminationStatus))"); break }
                }
            }
            sleep(5)
        }
    }
}

/// The launchd job that starts the watchdog at login (~/Library/LaunchAgents/io.github.maximianuu.farabi.watchdog.plist).
/// The "keep the engine running in the background" switch in Settings installs and removes it.
enum LaunchAgent {
    static let label = "io.github.maximianuu.farabi.watchdog"
    static var plistPath: String { NSHomeDirectory() + "/Library/LaunchAgents/\(label).plist" }
    static var domain: String { "gui/\(getuid())" }

    @discardableResult
    static func launchctl(_ args: [String]) -> Int32 {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        p.arguments = args
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        guard (try? p.run()) != nil else { return -1 }
        p.waitUntilExit()
        return p.terminationStatus
    }

    static var isEnabled: Bool { FileManager.default.fileExists(atPath: plistPath) && launchctl(["print", "\(domain)/\(label)"]) == 0 }

    /// Writes the job definition and loads it. Returns an error message, or nil on success.
    static func enable() -> String? {
        guard let binary = Bundle.main.executablePath else { return "could not read the app's executable path" }
        let definition: [String: Any] = [
            "Label": label,
            "ProgramArguments": [binary, "--watchdog"],
            "RunAtLoad": true,
            "KeepAlive": true,
            "AssociatedBundleIdentifiers": ["io.github.maximianuu.Farabi"],
            "ProcessType": "Interactive",
            "StandardOutPath": Watchdog.logPath,
            "StandardErrorPath": Watchdog.logPath,
        ]
        guard let data = try? PropertyListSerialization.data(fromPropertyList: definition, format: .xml, options: 0) else { return "could not serialize the job definition" }
        launchctl(["bootout", "\(domain)/\(label)"])
        do {
            try FileManager.default.createDirectory(atPath: (plistPath as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
            try data.write(to: URL(fileURLWithPath: plistPath))
        } catch { return error.localizedDescription }
        launchctl(["enable", "\(domain)/\(label)"])
        let status = launchctl(["bootstrap", domain, plistPath])
        if status != 0 && launchctl(["print", "\(domain)/\(label)"]) != 0 { return "could not load the launchd job (status \(status))" }
        return nil
    }

    static func disable() {
        launchctl(["bootout", "\(domain)/\(label)"])
        try? FileManager.default.removeItem(atPath: plistPath)
    }
}
