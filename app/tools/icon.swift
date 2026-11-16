// Farabi app icon: a white capital "F" on a rounded square with a teal -> deep blue gradient.
// Usage: swift tools/icon.swift Sources/Assets.xcassets/AppIcon.appiconset
import AppKit

let target = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "AppIcon.appiconset"
try? FileManager.default.createDirectory(atPath: target, withIntermediateDirectories: true)

func draw(_ side: Int) -> Data {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: side, pixelsHigh: side, bitsPerSample: 8, samplesPerPixel: 4,
                               hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    rep.size = NSSize(width: side, height: side)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let s = CGFloat(side)
    let inset = s * 0.085
    let square = NSRect(x: inset, y: inset, width: s - 2 * inset, height: s - 2 * inset)
    let path = NSBezierPath(roundedRect: square, xRadius: square.width * 0.225, yRadius: square.width * 0.225)
    NSGradient(colors: [NSColor(red: 15 / 255, green: 118 / 255, blue: 110 / 255, alpha: 1), NSColor(red: 30 / 255, green: 58 / 255, blue: 138 / 255, alpha: 1)])!
        .draw(in: path, angle: -60)
    let glyph = NSAttributedString(string: "F", attributes: [
        .font: NSFont.systemFont(ofSize: s * 0.56, weight: .semibold), .foregroundColor: NSColor.white])
    let box = glyph.size()
    glyph.draw(at: NSPoint(x: (s - box.width) / 2, y: (s - box.height) / 2 + s * 0.005))
    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])!
}

var images: [[String: String]] = []
for (points, scale) in [(16, 1), (16, 2), (32, 1), (32, 2), (128, 1), (128, 2), (256, 1), (256, 2), (512, 1), (512, 2)] {
    let name = "icon-\(points)@\(scale)x.png"
    try! draw(points * scale).write(to: URL(fileURLWithPath: target + "/" + name))
    images.append(["idiom": "mac", "size": "\(points)x\(points)", "scale": "\(scale)x", "filename": name])
}
let contents: [String: Any] = ["images": images, "info": ["version": 1, "author": "xcode"]]
try! JSONSerialization.data(withJSONObject: contents, options: [.prettyPrinted]).write(to: URL(fileURLWithPath: target + "/Contents.json"))
print("icon written:", target)
