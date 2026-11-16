import AppKit
import SwiftUI

// Farabi's UI: chats and engine status on the left, a plain chat in the middle (no system prompt, no tools, no connectors),
// and a Settings sheet with the "keep the engine running in the background" switch.

let farabiTeal = Color(red: 15 / 255, green: 118 / 255, blue: 110 / 255)

/// UI language: English by default, Turkish selectable in Settings. Changes apply immediately (views observe `AppState.language`).
enum Lang {
    static var code: String = UserDefaults.standard.string(forKey: "language") ?? "en"
    static var isTurkish: Bool { code == "tr" }
}

/// `tx("Türkçe", "English")` returns the string for the selected UI language.
/// The first argument is the Turkish text; it is the only place Turkish appears in the source.
func tx(_ tr: String, _ en: String) -> String { Lang.isTurkish ? tr : en }

struct Message: Identifiable, Equatable, Codable {
    var id = UUID()
    var role: String               // "user" | "assistant"
    var text: String
    var thinking = ""
    var stats: String?
    var error: String?
    var model: String?             // the model that produced the reply
}

/// A saved chat: ~/Library/Application Support/Farabi/chats/<id>.json (stays on this Mac; never sent anywhere).
struct Chat: Identifiable, Equatable, Codable {
    var id = UUID()
    var title: String
    var model: String              // last used model
    var date = Date()
    var messages: [Message] = []
}

enum ChatStore {
    static var folder: URL {
        let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("Farabi/chats", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    static func fileURL(_ id: UUID) -> URL { folder.appendingPathComponent(id.uuidString + ".json") }
    static func loadAll() -> [Chat] {
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "json" }
            .compactMap { (try? Data(contentsOf: $0)).flatMap { try? decoder.decode(Chat.self, from: $0) } }
            .sorted { $0.date > $1.date }
    }
    static func save(_ chat: Chat) {
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601; encoder.outputFormatting = [.prettyPrinted]
        if let data = try? encoder.encode(chat) { try? data.write(to: fileURL(chat.id), options: .atomic) }
    }
    static func delete(_ id: UUID) { try? FileManager.default.trashItem(at: fileURL(id), resultingItemURL: nil) }      // moves to the Trash
}

func gbText(_ bytes: Int64) -> String {
    let s = String(format: "%.1f GB", Double(bytes) / 1e9)
    return Lang.isTurkish ? s.replacingOccurrences(of: ".", with: ",") : s
}

@MainActor
final class AppState: ObservableObject {
    static let shared = AppState()
    @Published var engine = EngineStatus()
    @Published var models: [ModelInfo] = []
    @Published var selected: String? = UserDefaults.standard.string(forKey: "selectedModel") {
        didSet { UserDefaults.standard.set(selected, forKey: "selectedModel") }
    }
    @Published var chats: [Chat] = ChatStore.loadAll()
    @Published var selectedChat: UUID?
    @Published var isGenerating = false
    @Published var think = false
    @Published var background = false
    @Published var notice: String?
    @Published var engineBusy = false
    @Published var modelBusy: String?
    @Published var settingsOpen = false
    @Published var language: String = Lang.code {
        didSet { Lang.code = language; UserDefaults.standard.set(language, forKey: "language") }
    }
    private var task: Task<Void, Never>?

    var selectedModel: ModelInfo? { models.first { $0.id == selected } }
    var messages: [Message] { chats.first { $0.id == selectedChat }?.messages ?? [] }

    func refresh() async {
        let s = await Engine.status()
        if s != engine { engine = s }
        guard s.running else { return }
        let m = await Engine.models()
        if !m.isEmpty, m != models { models = m }
        if selected == nil || !models.contains(where: { $0.id == selected }) { selected = models.first?.id }
    }

    func startEngine() async {
        guard !engine.running, !engineBusy else { return }
        engineBusy = true
        defer { engineBusy = false }
        guard Engine.isInstalled else {
            notice = tx("Motor bulunamadı. Ayarlar'da gösterilen motor klasörünü ve Python yolunu kontrol et (engineFolder / enginePython).",
                        "Engine not found. Check the engine folder and Python path shown in Settings (engineFolder / enginePython).")
            return
        }
        if !LaunchAgent.isEnabled { Engine.startProcess() }          // with background mode on, the watchdog starts it
        for _ in 0..<60 {
            try? await Task.sleep(nanoseconds: 500_000_000)
            await refresh()
            if engine.running { notice = nil; return }
        }
        notice = tx("Motor başlamadı. Günlüğe bak: ", "The engine did not start. Check the log: ") + "~/Library/Logs/farabi-engine-process.log"
    }

    func stopEngine() async {
        guard engine.running, !engineBusy else { return }
        engineBusy = true
        defer { engineBusy = false }
        stopGenerating()
        Engine.stopProcess()
        for _ in 0..<20 {
            try? await Task.sleep(nanoseconds: 300_000_000)
            await refresh()
            if !engine.running { break }
        }
        if LaunchAgent.isEnabled {                       // the watchdog restarts it within a few seconds
            for _ in 0..<60 {
                try? await Task.sleep(nanoseconds: 500_000_000)
                await refresh()
                if engine.running { break }
            }
        }
    }

    func load(_ id: String) {
        modelBusy = id
        Task { await Engine.load(id); await refresh(); modelBusy = nil }
    }

    func unload(_ id: String) {
        modelBusy = id
        Task { await Engine.unload(id); await refresh(); modelBusy = nil }
    }

    func send(_ text: String) {
        guard let model = selected, !isGenerating else { return }
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        if selectedChat == nil || !chats.contains(where: { $0.id == selectedChat }) {
            let chat = Chat(title: String(t.replacingOccurrences(of: "\n", with: " ").prefix(48)), model: model)
            chats.insert(chat, at: 0)
            selectedChat = chat.id
        }
        guard let chatID = selectedChat, let index = chats.firstIndex(where: { $0.id == chatID }) else { return }
        chats[index].messages.append(Message(role: "user", text: t))
        let outgoing = chats[index].messages.map { ["role": $0.role, "content": $0.text] }
        chats[index].messages.append(Message(role: "assistant", text: "", model: model))
        chats[index].model = model
        chats[index].date = Date()
        isGenerating = true
        let think = self.think
        task = Task { [weak self] in
            @MainActor func update(_ change: (inout Message) -> Void) {
                guard let self, let k = self.chats.firstIndex(where: { $0.id == chatID }), !self.chats[k].messages.isEmpty else { return }
                change(&self.chats[k].messages[self.chats[k].messages.count - 1])
            }
            do {
                for try await chunk in Engine.chat(model: model, messages: outgoing, think: think) {
                    update { last in
                        last.text += chunk.content
                        last.thinking += chunk.thinking
                        if let e = chunk.error { last.error = e }
                        if chunk.done { last.stats = chunk.stats }
                    }
                }
            } catch {
                let cancelled = error is CancellationError || (error as? URLError)?.code == .cancelled
                if !cancelled { update { $0.error = error.localizedDescription } }
            }
            if let self {
                self.isGenerating = false
                if let c = self.chats.first(where: { $0.id == chatID }) { ChatStore.save(c) }
                await self.refresh()
            }
        }
    }

    func stopGenerating() {
        task?.cancel()
        task = nil
        isGenerating = false
        if let c = chats.first(where: { $0.id == selectedChat }) { ChatStore.save(c) }
    }

    /// Blank page: the chat is created and saved with the first message.
    func newChat() {
        guard !isGenerating else { return }
        selectedChat = nil
    }

    func selectChat(_ id: UUID) {
        guard !isGenerating else { return }
        selectedChat = id
        if let m = chats.first(where: { $0.id == id })?.model, models.contains(where: { $0.id == m }) { selected = m }
    }

    func deleteChat(_ id: UUID) {
        if selectedChat == id { stopGenerating(); selectedChat = nil }
        chats.removeAll { $0.id == id }
        ChatStore.delete(id)
    }

    func setBackground(_ on: Bool) {
        if on {
            if let e = LaunchAgent.enable() { notice = tx("Arka plan açılamadı: ", "Could not enable background mode: ") + e }
        } else {
            LaunchAgent.disable()                        // an engine started by the watchdog stops with it
            Task {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                await refresh()
                if !engine.running { await startEngine() }   // keep the engine up while the app is open
            }
        }
        background = LaunchAgent.isEnabled
    }
}

// MARK: Sidebar

struct ChatRow: View {
    let chat: Chat
    @ObservedObject var state = AppState.shared
    @State private var hovering = false
    var isSelected: Bool { state.selectedChat == chat.id }
    var body: some View {
        HStack(spacing: 6) {
            Text(chat.title).lineLimit(1)
            Spacer(minLength: 4)
            if hovering {
                Button { state.deleteChat(chat.id) } label: { Image(systemName: "trash").font(.caption) }
                    .buttonStyle(.plain).foregroundStyle(.secondary).help(tx("Sohbeti sil (Çöp'e taşır)", "Delete chat (moves to Trash)"))
            }
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: 8).fill(isSelected ? farabiTeal.opacity(0.22) : (hovering ? Color.primary.opacity(0.06) : .clear)))
        .contentShape(Rectangle())
        .onTapGesture { state.selectChat(chat.id) }
        .onHover { hovering = $0 }
        .contextMenu { Button(tx("Sohbeti sil", "Delete chat")) { state.deleteChat(chat.id) } }
        .help(chat.title)
    }
}

struct Sidebar: View {
    @ObservedObject var state = AppState.shared

    private func header(_ t: String) -> some View {
        Text(t).font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.horizontal, 10).padding(.top, 14).padding(.bottom, 2)
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    Button { state.newChat() } label: {
                        Label(tx("Yeni sohbet", "New chat"), systemImage: "square.and.pencil").frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 10).padding(.vertical, 7).contentShape(Rectangle())
                    }
                    .buttonStyle(.plain).padding(.top, 8)
                    header(tx("SOHBETLER", "CHATS"))
                    ForEach(state.chats) { chat in ChatRow(chat: chat) }
                    if state.chats.isEmpty {
                        Text(tx("Henüz sohbet yok", "No chats yet")).font(.callout).foregroundStyle(.tertiary).padding(.horizontal, 10).padding(.vertical, 6)
                    }
                }
                .padding(.horizontal, 8).padding(.bottom, 12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Divider()
            // Engine status, bottom left above Settings
            VStack(alignment: .leading, spacing: 7) {
                HStack(spacing: 8) {
                    Circle().fill(state.engine.running ? Color.green : Color.red).frame(width: 8, height: 8)
                    Text(state.engine.running ? tx("Motor çalışıyor", "Engine running") : tx("Motor kapalı", "Engine is off")).font(.callout)
                    Spacer()
                    if state.engineBusy {
                        ProgressView().controlSize(.small)
                    } else if state.engine.running {
                        Button(state.background ? tx("Yeniden başlat", "Restart") : tx("Durdur", "Stop")) { Task { await state.stopEngine() } }.controlSize(.mini)
                    } else {
                        Button(tx("Başlat", "Start")) { Task { await state.startEngine() } }.controlSize(.mini)
                    }
                }
                if state.engine.running {
                    Label(state.engine.networkLock ? tx("Ağa kapalı: yalnız bu Mac", "Offline: this Mac only") : tx("Ağ kilidi KAPALI", "Network lock is OFF"),
                          systemImage: state.engine.networkLock ? "lock.fill" : "lock.open")
                        .font(.caption).foregroundStyle(state.engine.networkLock ? Color.secondary : Color.orange)
                }
                Label(state.background ? tx("Arka planda çalışır", "Runs in the background") : tx("Yalnız uygulama açıkken", "Only while the app is open"),
                      systemImage: state.background ? "moon.zzz.fill" : "macwindow")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16).padding(.vertical, 10)
            Divider()
            // Bottom left: Settings
            Button { state.settingsOpen = true } label: {
                HStack(spacing: 8) {
                    Text("F").font(.system(size: 15, weight: .semibold, design: .serif)).foregroundStyle(farabiTeal)
                    Text("Farabi").fontWeight(.medium)
                    Text("·").foregroundStyle(.tertiary)
                    Text(tx("Ayarlar", "Settings")).foregroundStyle(.secondary)
                    Spacer()
                    Image(systemName: "gearshape").foregroundStyle(.secondary)
                }
                .padding(.horizontal, 16).padding(.vertical, 11).contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help(tx("Ayarlar (⌘,)", "Settings (⌘,)"))
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

/// Model picker in the toolbar: model names, size, memory state; load / unload.
struct ModelPicker: View {
    @ObservedObject var state = AppState.shared
    var body: some View {
        Menu {
            ForEach(state.models) { m in
                Button { state.selected = m.id } label: {
                    Text((state.selected == m.id ? "✓ " : "   ") + m.id + "   " + [m.size, m.quantization, gbText(m.diskBytes)].filter { !$0.isEmpty }.joined(separator: " · ")
                         + (m.isLoaded ? "   ● " + tx("bellekte ", "in memory ") + gbText(m.memoryBytes) : ""))
                }
            }
            Divider()
            ForEach(state.models) { m in
                if m.isLoaded {
                    Button(tx("Bellekten boşalt: ", "Unload from memory: ") + m.id) { state.unload(m.id) }
                } else {
                    Button(tx("Belleğe yükle: ", "Load into memory: ") + m.id) { state.load(m.id) }
                }
            }
        } label: {
            HStack(spacing: 6) {
                if state.modelBusy != nil {
                    ProgressView().controlSize(.small)
                } else {
                    Circle().fill(state.selectedModel?.isLoaded == true ? Color.green : Color.secondary.opacity(0.45)).frame(width: 7, height: 7)
                }
                Text(state.selectedModel?.id ?? tx("Model seç", "Pick a model")).font(.system(.body, design: .monospaced))
                if let m = state.selectedModel, m.isLoaded { Text(gbText(m.memoryBytes)).font(.caption).foregroundStyle(.secondary) }
            }
            .padding(.horizontal, 4)
        }
        .menuIndicator(.visible)
        .disabled(state.models.isEmpty)
        .help(tx("Model seç · bellekteki model yeşil noktayla gösterilir", "Choose a model · the green dot means it is in memory"))
    }
}

// MARK: Chat

struct MessageView: View {
    let message: Message
    let isWriting: Bool
    var body: some View {
        if message.role == "user" {
            HStack {
                Spacer(minLength: 60)
                Text(message.text).textSelection(.enabled)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(farabiTeal.opacity(0.16), in: RoundedRectangle(cornerRadius: 12))
            }
        } else {
            VStack(alignment: .leading, spacing: 6) {
                if !message.thinking.isEmpty {
                    DisclosureGroup(tx("Düşünce", "Thinking")) {
                        Text(message.thinking).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .font(.callout).foregroundStyle(.secondary)
                }
                if message.text.isEmpty && message.error == nil && isWriting {
                    HStack(spacing: 8) { ProgressView().controlSize(.small); Text(tx("Yanıt bekleniyor…", "Waiting for the reply…")).foregroundStyle(.secondary) }
                } else {
                    Text(message.text).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                }
                if let e = message.error { Label(e, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.red) }
                if let s = message.stats { Text([message.model, s].compactMap { $0 }.joined(separator: " · ")).font(.caption).foregroundStyle(.tertiary) }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

struct ChatView: View {
    @ObservedObject var state = AppState.shared
    @State private var draft = ""
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 0) {
            if !state.engine.running {
                offlineView
            } else if state.messages.isEmpty {
                emptyView
            } else {
                ScrollViewReader { scroller in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 16) {
                            ForEach(state.messages) { m in
                                MessageView(message: m, isWriting: state.isGenerating && m.id == state.messages.last?.id).id(m.id)
                            }
                            Color.clear.frame(height: 1).id("bottom")
                        }
                        .padding(20)
                        .frame(maxWidth: 820)
                        .frame(maxWidth: .infinity)
                    }
                    .onChange(of: state.messages.last?.text) { scroller.scrollTo("bottom", anchor: .bottom) }
                    .onChange(of: state.messages.count) { scroller.scrollTo("bottom", anchor: .bottom) }
                }
            }
            if let n = state.notice {
                Text(n).font(.callout).foregroundStyle(.orange).padding(.horizontal, 20).padding(.bottom, 6)
            }
            inputBar
        }
        .navigationTitle("Farabi")
        .navigationSubtitle(tx("sade sohbet", "plain chat"))
        .toolbar {
            ToolbarItem(placement: .principal) {
                HStack(spacing: 6) {
                    ModelPicker()
                    // Eject button: unloads the selected model from memory
                    if let m = state.selectedModel, m.isLoaded, state.modelBusy == nil {
                        Button { state.unload(m.id) } label: { Image(systemName: "eject.fill") }
                            .help(tx("Modeli bellekten çıkar", "Eject the model from memory"))
                            .disabled(state.isGenerating)
                    }
                }
            }
            ToolbarItem(placement: .primaryAction) {
                Button { state.newChat() } label: { Label(tx("Yeni sohbet", "New chat"), systemImage: "square.and.pencil") }
                    .help(tx("Yeni sohbet (⌘N)", "New chat (⌘N)")).disabled(state.selectedChat == nil || state.isGenerating)
            }
        }
        .onAppear { focused = true }
    }

    private var offlineView: some View {
        VStack(spacing: 12) {
            Image(systemName: "powersleep").font(.system(size: 34)).foregroundStyle(.secondary)
            Text(tx("Motor kapalı", "Engine is off")).font(.title3)
            if state.engineBusy { ProgressView(tx("Başlatılıyor…", "Starting…")).controlSize(.small) } else {
                Button(tx("Motoru başlat", "Start the engine")) { Task { await state.startEngine() } }.buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyView: some View {
        VStack(spacing: 8) {
            Text("F").font(.system(size: 56, weight: .semibold, design: .serif)).foregroundStyle(farabiTeal)
            Text(state.selectedModel?.id ?? tx("Model seç", "Pick a model")).font(.system(.title3, design: .monospaced))
            Text(tx("Sade sohbet: sistem istemi, araç ya da bağlayıcı yok.\nModel ne ise o; yazdığın olduğu gibi gider.", "Plain chat: no system prompt, tools or connectors.\nThe model as it is; what you type is sent as is."))
                .multilineTextAlignment(.center).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 10) {
            VStack(alignment: .leading, spacing: 6) {
                TextField(tx("Mesaj yaz", "Message"), text: $draft, axis: .vertical)
                    .textFieldStyle(.plain).lineLimit(1...8).focused($focused)
                    .onSubmit { submit() }
                    .disabled(!state.engine.running || state.selected == nil)
                Toggle(tx("Düşünme", "Thinking"), isOn: $state.think).toggleStyle(.checkbox).font(.caption).foregroundStyle(.secondary)
                    .help(tx("Açıkken model yanıttan önce düşünür; düşüncesi ayrı gösterilir", "When on, the model thinks before answering; its thinking is shown separately"))
            }
            if state.isGenerating {
                Button { state.stopGenerating() } label: { Image(systemName: "stop.circle.fill").font(.system(size: 24)) }
                    .buttonStyle(.plain).foregroundStyle(.secondary).help(tx("Durdur", "Stop"))
            } else {
                Button { submit() } label: { Image(systemName: "arrow.up.circle.fill").font(.system(size: 24)) }
                    .buttonStyle(.plain).foregroundStyle(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.secondary : farabiTeal)
                    .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !state.engine.running || state.selected == nil)
                    .help(tx("Gönder (↩)", "Send (↩)"))
            }
        }
        .padding(12)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 14))
        .frame(maxWidth: 820)
        .padding(.horizontal, 20).padding(.bottom, 16).padding(.top, 6)
    }

    private func submit() {
        let t = draft
        guard !t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !state.isGenerating else { return }
        draft = ""
        state.send(t)
    }
}

struct MainView: View {
    @ObservedObject var state = AppState.shared
    @AppStorage("sidebarOpen") private var sidebarOpen = true
    var body: some View {
        // Plain layout (not NavigationSplitView): a flat sidebar background instead of the translucent system material
        HStack(spacing: 0) {
            if sidebarOpen {
                Sidebar().frame(width: 270)
                Divider()
            }
            ChatView()
        }
        .sheet(isPresented: $state.settingsOpen) { SettingsView(isSheet: true) }
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button { sidebarOpen.toggle() } label: { Image(systemName: "sidebar.left") }.help(tx("Sohbetler sütununu göster / gizle", "Show / hide the chats column"))
            }
        }
        .tint(farabiTeal)
        .task {
            while !Task.isCancelled {
                await state.refresh()
                try? await Task.sleep(nanoseconds: 3_000_000_000)
            }
        }
    }
}

// MARK: Settings

struct SettingsView: View {
    @ObservedObject var state = AppState.shared
    var isSheet = false                                 // opened inside the window: show a title and a Done button
    var body: some View {
        VStack(spacing: 0) {
            if isSheet {
                HStack {
                    Text(tx("Ayarlar", "Settings")).font(.headline)
                    Spacer()
                    Button(tx("Kapat", "Done")) { state.settingsOpen = false }.keyboardShortcut(.defaultAction)
                }
                .padding(.horizontal, 20).padding(.top, 16).padding(.bottom, 4)
            }
            form
        }
        .frame(width: 560)
        .tint(farabiTeal)
        .onAppear { state.background = LaunchAgent.isEnabled }
    }

    private var form: some View {
        Form {
            Section {
                Picker(tx("Arayüz dili", "Interface language"), selection: $state.language) {
                    Text("English").tag("en")
                    Text("Türkçe").tag("tr")
                }
            }
            Section {
                Toggle(isOn: Binding(get: { state.background }, set: { state.setBackground($0) })) {
                    Text(tx("Uygulama kapalıyken de arka planda çalış", "Keep running in the background after the app is quit"))
                    Text(tx("Açıkken motor Mac açılınca kendiliğinden başlar, Farabi'den çıksan da çalışır, çökerse yeniden başlatılır. Kapalıyken Farabi'den çıkınca motor da durur (o an bir yanıt üretmiyorsa).", "When on, the engine starts when the Mac starts, keeps running after you quit Farabi, and is restarted if it crashes. When off, quitting Farabi also stops the engine (unless it is generating a reply)."))
                }
            }
            Section(tx("Motor", "Engine")) {
                LabeledContent(tx("Kod klasörü", "Code folder")) { Text(Engine.folder).font(.system(.callout, design: .monospaced)).textSelection(.enabled) }
                LabeledContent("Python") { Text(Engine.python).font(.system(.callout, design: .monospaced)).textSelection(.enabled).lineLimit(2) }
                LabeledContent(tx("Adres", "Address")) { Text("127.0.0.1:\(Engine.port) " + tx("(yalnız bu Mac)", "(this Mac only)")).font(.system(.callout, design: .monospaced)) }
                HStack {
                    Button(tx("İstek günlüğünü göster", "Show request log")) { NSWorkspace.shared.selectFile(Engine.requestLog, inFileViewerRootedAtPath: "") }
                    Button(tx("Bekçi günlüğünü göster", "Show watchdog log")) { NSWorkspace.shared.selectFile(Watchdog.logPath, inFileViewerRootedAtPath: "") }
                }
            }
        }
        .formStyle(.grouped)
    }
}

// MARK: App

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // First launch from /Applications: background mode defaults to ON so the engine does not depend on the window being open.
        let defaults = UserDefaults.standard
        if !defaults.bool(forKey: "firstRunDone"), Bundle.main.bundlePath.hasPrefix("/Applications/") {
            defaults.set(true, forKey: "firstRunDone")
            _ = LaunchAgent.enable()
        }
        Task { @MainActor in
            let state = AppState.shared
            state.background = LaunchAgent.isEnabled
            await state.refresh()
            if !state.engine.running { await state.startEngine() }
        }
        TestHook.register()
    }

    func applicationWillTerminate(_ notification: Notification) {
        guard !LaunchAgent.isEnabled else { return }     // background mode on: the engine keeps running
        let s = Engine.probeSync()
        if s.running && !s.busy { Engine.stopProcess() }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool { true }
}

struct FarabiApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @ObservedObject private var state = AppState.shared    // menu titles refresh when the language changes
    var body: some Scene {
        WindowGroup("Farabi") {
            MainView().frame(minWidth: 820, minHeight: 520)
        }
        .defaultSize(width: 1040, height: 680)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button(tx("Yeni Sohbet", "New Chat")) { AppState.shared.newChat() }.keyboardShortcut("n")
            }
            CommandGroup(replacing: .appSettings) {
                Button(tx("Ayarlar…", "Settings…")) { AppState.shared.settingsOpen = true }.keyboardShortcut(",")
            }
        }
    }
}

// MARK: Test hook (verify the UI without mouse automation)

/// `defaults write io.github.maximianuu.Farabi test "<png>|<command>"`, then post the distributed notification
/// `io.github.maximianuu.Farabi.test`. The window is saved to `<png>` and a state dump to `<png>.json` (png `-` = no screenshot).
/// Commands: empty (screenshot only) · `settings` · `settings-close` · `model:<name>` · `ask:<text>` · `new` · `chat:<n>` ·
/// `chat-delete:<n>` · `think:1|0` · `lang:tr|en` · `background:1|0` · `unload`.
enum TestHook {
    static let domain = "io.github.maximianuu.Farabi"

    static func register() {
        DistributedNotificationCenter.default().addObserver(forName: .init(domain + ".test"), object: nil, queue: .main) { _ in
            Task { @MainActor in await run() }
        }
    }

    @MainActor static func run() async {
        CFPreferencesAppSynchronize(domain as CFString)
        let defaults = UserDefaults.standard
        guard let raw = defaults.string(forKey: "test") else { return }
        defaults.removeObject(forKey: "test")
        let parts = raw.split(separator: "|", maxSplits: 1, omittingEmptySubsequences: false).map(String.init)
        let path = parts[0], command = parts.count > 1 ? parts[1] : ""
        let state = AppState.shared
        var wait: UInt64 = 1_200_000_000
        if command == "settings" {
            state.settingsOpen = true
        } else if command.hasPrefix("model:") {
            let name = String(command.dropFirst(6))
            state.selected = state.models.first { $0.id == name }?.id ?? state.selected
        } else if command.hasPrefix("ask:") {
            state.send(String(command.dropFirst(4)))
            wait = 400_000_000
            for _ in 0..<600 where state.isGenerating { try? await Task.sleep(nanoseconds: 300_000_000) }
        } else if command.hasPrefix("lang:") {
            state.language = String(command.dropFirst(5))
        } else if command == "settings-close" {
            state.settingsOpen = false
        } else if command == "new" {
            state.newChat()
        } else if command.hasPrefix("chat:") {            // select by position (0 = newest)
            if let n = Int(command.dropFirst(5)), state.chats.indices.contains(n) { state.selectChat(state.chats[n].id) }
        } else if command.hasPrefix("chat-delete:") {
            if let n = Int(command.dropFirst(12)), state.chats.indices.contains(n) { state.deleteChat(state.chats[n].id) }
        } else if command.hasPrefix("think:") {
            state.think = command.hasSuffix("1")
        } else if command.hasPrefix("background:") {
            state.setBackground(command.hasSuffix("1"))
            wait = 4_000_000_000
        } else if command == "unload" {
            for m in state.models where m.isLoaded { await Engine.unload(m.id) }
            await state.refresh()
        }
        try? await Task.sleep(nanoseconds: wait)
        guard path != "-" else { return }
        let windows = NSApp.windows.filter { $0.isVisible && $0.styleMask.contains(.titled) }
        let target = command == "settings" ? (windows.first { $0.isSheet } ?? NSApp.windows.first { $0.isSheet && $0.isVisible } ?? windows.max { $0.frame.width < $1.frame.width })
            : windows.filter { !$0.isSheet }.max { $0.frame.width < $1.frame.width }
        guard let w = target, let v = w.contentView?.superview ?? w.contentView, let rep = v.bitmapImageRepForCachingDisplay(in: v.bounds) else { return }
        v.cacheDisplay(in: v.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
        let dump: [String: Any] = [
            "engine": ["running": state.engine.running, "version": state.engine.version, "networkLock": state.engine.networkLock],
            "background": state.background, "selected": state.selected ?? "", "generating": state.isGenerating, "language": state.language,
            "chats": state.chats.map { ["title": $0.title, "model": $0.model, "messages": "\($0.messages.count)", "selected": $0.id == state.selectedChat ? "yes" : ""] },
            "models": state.models.map { ["name": $0.id, "size": $0.size, "quantization": $0.quantization, "disk": gbText($0.diskBytes), "memory": $0.isLoaded ? gbText($0.memoryBytes) : ""] },
            "messages": state.messages.map { ["role": $0.role, "text": $0.text, "thinking": String($0.thinking.prefix(200)), "stats": $0.stats ?? "", "error": $0.error ?? ""] },
        ]
        try? JSONSerialization.data(withJSONObject: dump, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: path + ".json"))
    }
}
