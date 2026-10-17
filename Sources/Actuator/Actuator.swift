import Accelerate
import AppKit
import ApplicationServices
import CoreGraphics
import Foundation
import ImageIO

public enum GUIAction: Equatable, CustomStringConvertible {
    case click(x: Int, y: Int)
    case doubleClick(x: Int, y: Int)
    case rightClick(x: Int, y: Int)
    case move(x: Int, y: Int)
    case drag(x1: Int, y1: Int, x2: Int, y2: Int)
    case typeText(String)
    case key(String)
    case scroll(dx: Int, dy: Int)
    case wait(ms: Int)
    case done

    public var description: String {
        switch self {
        case let .click(x, y): return "click(\(x), \(y))"
        case let .doubleClick(x, y): return "double_click(\(x), \(y))"
        case let .rightClick(x, y): return "right_click(\(x), \(y))"
        case let .move(x, y): return "move(\(x), \(y))"
        case let .drag(x1, y1, x2, y2): return "drag(\(x1), \(y1), \(x2), \(y2))"
        case let .typeText(s): return "type(\"\(s)\")"
        case let .key(k): return "key(\"\(k)\")"
        case let .scroll(dx, dy): return "scroll(\(dx), \(dy))"
        case let .wait(ms): return "wait(\(ms))"
        case .done: return "done()"
        }
    }

    public var coords: (x: Int, y: Int)? {
        switch self {
        case let .click(x, y), let .doubleClick(x, y), let .rightClick(x, y), let .move(x, y):
            return (x, y)
        case let .drag(x1, y1, _, _):
            return (x1, y1)
        default:
            return nil
        }
    }
}

public enum ActionParser {
    public static func parse(_ raw: String) -> GUIAction? {
        let s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let openIdx = s.firstIndex(of: "("),
              let closeIdx = s[openIdx...].firstIndex(of: ")")
        else { return nil }
        let fn = String(s[..<openIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .whitespacesAndNewlines).last ?? ""
        let inside = String(s[s.index(after: openIdx)..<closeIdx]).trimmingCharacters(in: .whitespacesAndNewlines)

        func coords(_ count: Int) -> [Int]? {
            let parts = inside.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            guard parts.count >= count else { return nil }
            var res: [Int] = []
            for i in 0..<parts.count {
                if let v = Int(parts[i]) {
                    res.append(v)
                } else if let d = Double(parts[i]) {
                    res.append(d > 0.0 && d <= 1.0 ? Int((d * 1000.0).rounded()) : Int(d.rounded()))
                } else {
                    return nil
                }
            }
            return res
        }

        func point2() -> (Int, Int)? {
            guard let v = coords(2) else { return nil }
            if v.count >= 4 {
                return ((v[0] + v[2]) / 2, (v[1] + v[3]) / 2)
            }
            return (v[0], v[1])
        }

        func quoted() -> String {
            if inside.count >= 2 && ((inside.first == "\"" && inside.last == "\"") || (inside.first == "'" && inside.last == "'")) {
                return String(inside.dropFirst().dropLast())
            }
            return inside
        }

        switch fn {
        case "click":
            guard let (x, y) = point2() else { return nil }
            return .click(x: x, y: y)
        case "double_click":
            guard let (x, y) = point2() else { return nil }
            return .doubleClick(x: x, y: y)
        case "right_click":
            guard let (x, y) = point2() else { return nil }
            return .rightClick(x: x, y: y)
        case "move":
            guard let (x, y) = point2() else { return nil }
            return .move(x: x, y: y)
        case "drag":
            guard let v = coords(4) else { return nil }
            return .drag(x1: v[0], y1: v[1], x2: v[2], y2: v[3])
        case "type":
            return .typeText(quoted())
        case "key":
            return .key(quoted())
        case "scroll":
            guard let v = coords(2) else { return nil }
            return .scroll(dx: v[0], dy: v[1])
        case "wait":
            guard let v = coords(1) else { return nil }
            return .wait(ms: v[0])
        case "done":
            return .done
        default:
            return nil
        }
    }
}

public enum ScreenCapture {
    /// Loads a PNG/image file from disk into tightly packed RGB8 bytes `[targetH, targetW, 3]`.
    /// When the image dimensions already match `(targetW, targetH)`, extracts raw pixel bytes directly
    /// from the CGImage data provider to avoid any ColorSync ICC profile shift.
    public static func loadRGB(path: String, targetW: Int = 640, targetH: Int = 384) throws -> [UInt8] {
        let url = URL(fileURLWithPath: path)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            throw NSError(domain: "ScreenCapture", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot load image \(path)"])
        }
        return extractRGB(from: cgImg, targetW: targetW, targetH: targetH)
    }

    /// Captures the main display (or a specific rect in screen points) into RGB8 bytes `[targetH, targetW, 3]`.
    public static func captureDisplayRGB(rect: CGRect? = nil, targetW: Int = 640, targetH: Int = 384) throws -> ([UInt8], CGRect) {
        let bounds = rect ?? CGDisplayBounds(CGMainDisplayID())
        typealias CreateImageFn = @convention(c) (CGRect, UInt32, UInt32, UInt32) -> Unmanaged<CGImage>?
        let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "CGWindowListCreateImage")
        guard let sym,
              let cgImg = unsafeBitCast(sym, to: CreateImageFn.self)(bounds, 1, 0, 0)?.takeRetainedValue()
        else {
            throw NSError(domain: "ScreenCapture", code: 2, userInfo: [NSLocalizedDescriptionKey: "Screen capture failed"])
        }
        let rgb = extractRGB(from: cgImg, targetW: targetW, targetH: targetH)
        return (rgb, bounds)
    }

    public static func extractRGB(from cgImg: CGImage, targetW: Int = 640, targetH: Int = 384) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: targetW * targetH * 3)
        if cgImg.width == targetW && cgImg.height == targetH,
           cgImg.bitsPerComponent == 8,
           let provider = cgImg.dataProvider,
           let cfData = provider.data {
            let ptr = CFDataGetBytePtr(cfData)!
            let bpr = cgImg.bytesPerRow
            let bpp = cgImg.bitsPerPixel / 8
            let alphaInfo = cgImg.alphaInfo
            let byteOrder = cgImg.bitmapInfo.intersection(.byteOrderMask)
            let isBGRA = (byteOrder == .byteOrder32Little) && (alphaInfo == .premultipliedFirst || alphaInfo == .first || alphaInfo == .noneSkipFirst)

            if bpp == 3 || bpp == 4 {
                for y in 0..<targetH {
                    let row = ptr + y * bpr
                    let dstBase = y * targetW * 3
                    for x in 0..<targetW {
                        let px = row + x * bpp
                        if isBGRA {
                            out[dstBase + x * 3 + 0] = px[2]
                            out[dstBase + x * 3 + 1] = px[1]
                            out[dstBase + x * 3 + 2] = px[0]
                        } else if bpp == 4 && (alphaInfo == .premultipliedFirst || alphaInfo == .first || alphaInfo == .noneSkipFirst) {
                            out[dstBase + x * 3 + 0] = px[1]
                            out[dstBase + x * 3 + 1] = px[2]
                            out[dstBase + x * 3 + 2] = px[3]
                        } else {
                            out[dstBase + x * 3 + 0] = px[0]
                            out[dstBase + x * 3 + 1] = px[1]
                            out[dstBase + x * 3 + 2] = px[2]
                        }
                    }
                }
                return out
            }
        }

        // Fallback / resize path via CGBitmapContext
        var rgba = [UInt8](repeating: 0, count: targetW * targetH * 4)
        let cs = cgImg.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        rgba.withUnsafeMutableBytes { raw in
            if let ctx = CGContext(
                data: raw.baseAddress,
                width: targetW,
                height: targetH,
                bitsPerComponent: 8,
                bytesPerRow: targetW * 4,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) {
                ctx.interpolationQuality = .medium
                ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: targetW, height: targetH))
            }
        }
        for i in 0..<(targetW * targetH) {
            out[i * 3 + 0] = rgba[i * 4 + 0]
            out[i * 3 + 1] = rgba[i * 4 + 1]
            out[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return out
    }
}

public final class Actuator {
    public nonisolated(unsafe) static var panicTriggered = false

    /// Maps normalized [0, 1000] x [0, 1000] Qwen3-VL coordinates into screen point coordinates within `bounds`.
    public static func mapToScreen(nx: Int, ny: Int, bounds: CGRect) -> CGPoint {
        let cx = max(0, min(1000, nx))
        let cy = max(0, min(1000, ny))
        let x = bounds.origin.x + ( CGFloat(cx) / 1000.0 ) * bounds.size.width
        let y = bounds.origin.y + ( CGFloat(cy) / 1000.0 ) * bounds.size.height
        return CGPoint(x: x, y: y)
    }

    /// Executes a parsed GUIAction directly on macOS via CoreGraphics CGEvent.
    public static func perform(_ action: GUIAction, in bounds: CGRect, dryRun: Bool = false) {
        if panicTriggered || dryRun { return }
        let src = CGEventSource(stateID: .hidSystemState)

        switch action {
        case let .move(nx, ny):
            let pt = mapToScreen(nx: nx, ny: ny, bounds: bounds)
            CGEvent(mouseEventSource: src, mouseType: .mouseMoved, mouseCursorPosition: pt, mouseButton: .left)?.post(tap: .cghidEventTap)

        case let .click(nx, ny):
            let pt = mapToScreen(nx: nx, ny: ny, bounds: bounds)
            CGEvent(mouseEventSource: src, mouseType: .mouseMoved, mouseCursorPosition: pt, mouseButton: .left)?.post(tap: .cghidEventTap)
            usleep(2000)
            CGEvent(mouseEventSource: src, mouseType: .leftMouseDown, mouseCursorPosition: pt, mouseButton: .left)?.post(tap: .cghidEventTap)
            usleep(4000)
            CGEvent(mouseEventSource: src, mouseType: .leftMouseUp, mouseCursorPosition: pt, mouseButton: .left)?.post(tap: .cghidEventTap)

        case let .doubleClick(nx, ny):
            let pt = mapToScreen(nx: nx, ny: ny, bounds: bounds)
            for clickNum in 1...2 {
                let down = CGEvent(mouseEventSource: src, mouseType: .leftMouseDown, mouseCursorPosition: pt, mouseButton: .left)
                down?.setIntegerValueField(.mouseEventClickState, value: Int64(clickNum))
                down?.post(tap: .cghidEventTap)
                usleep(3000)
                let up = CGEvent(mouseEventSource: src, mouseType: .leftMouseUp, mouseCursorPosition: pt, mouseButton: .left)
                up?.setIntegerValueField(.mouseEventClickState, value: Int64(clickNum))
                up?.post(tap: .cghidEventTap)
                usleep(3000)
            }

        case let .rightClick(nx, ny):
            let pt = mapToScreen(nx: nx, ny: ny, bounds: bounds)
            CGEvent(mouseEventSource: src, mouseType: .rightMouseDown, mouseCursorPosition: pt, mouseButton: .right)?.post(tap: .cghidEventTap)
            usleep(4000)
            CGEvent(mouseEventSource: src, mouseType: .rightMouseUp, mouseCursorPosition: pt, mouseButton: .right)?.post(tap: .cghidEventTap)

        case let .drag(nx1, ny1, nx2, ny2):
            let p1 = mapToScreen(nx: nx1, ny: ny1, bounds: bounds)
            let p2 = mapToScreen(nx: nx2, ny: ny2, bounds: bounds)
            CGEvent(mouseEventSource: src, mouseType: .leftMouseDown, mouseCursorPosition: p1, mouseButton: .left)?.post(tap: .cghidEventTap)
            usleep(8000)
            CGEvent(mouseEventSource: src, mouseType: .leftMouseDragged, mouseCursorPosition: p2, mouseButton: .left)?.post(tap: .cghidEventTap)
            usleep(8000)
            CGEvent(mouseEventSource: src, mouseType: .leftMouseUp, mouseCursorPosition: p2, mouseButton: .left)?.post(tap: .cghidEventTap)

        case let .typeText(text):
            let utf16 = Array(text.utf16)
            utf16.withUnsafeBufferPointer { buf in
                let down = CGEvent(keyboardEventSource: src, virtualKey: 0, keyDown: true)
                down?.keyboardSetUnicodeString(stringLength: buf.count, unicodeString: buf.baseAddress!)
                down?.post(tap: .cghidEventTap)
                let up = CGEvent(keyboardEventSource: src, virtualKey: 0, keyDown: false)
                up?.keyboardSetUnicodeString(stringLength: buf.count, unicodeString: buf.baseAddress!)
                up?.post(tap: .cghidEventTap)
            }

        case let .key(combo):
            postKeyCombo(combo, src: src)

        case let .scroll(dx, dy):
            CGEvent(scrollWheelEvent2Source: src, units: .pixel, wheelCount: 2, wheel1: Int32(dy), wheel2: Int32(dx), wheel3: 0)?.post(tap: .cghidEventTap)

        case let .wait(ms):
            usleep(UInt32(max(0, min(10_000, ms)) * 1000))

        case .done:
            break
        }
    }

    private static func postKeyCombo(_ combo: String, src: CGEventSource?) {
        let parts = combo.lowercased().split(separator: "+").map { String($0).trimmingCharacters(in: .whitespaces) }
        var flags: CGEventFlags = []
        var keyName = ""
        for p in parts {
            switch p {
            case "cmd", "command", "meta": flags.insert(.maskCommand)
            case "shift": flags.insert(.maskShift)
            case "alt", "option", "opt": flags.insert(.maskAlternate)
            case "ctrl", "control": flags.insert(.maskControl)
            default: keyName = p
            }
        }
        let keyCode = virtualKeyCode(for: keyName)
        let down = CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: true)
        down?.flags = flags
        down?.post(tap: .cghidEventTap)
        usleep(3000)
        let up = CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: false)
        up?.flags = flags
        up?.post(tap: .cghidEventTap)
    }

    private static func virtualKeyCode(for name: String) -> CGKeyCode {
        switch name {
        case "return", "enter": return 36
        case "tab": return 48
        case "space": return 49
        case "delete", "backspace": return 51
        case "escape", "esc": return 53
        case "left": return 123
        case "right": return 124
        case "down": return 125
        case "up": return 126
        case "a": return 0
        case "s": return 1
        case "d": return 2
        case "f": return 3
        case "c": return 8
        case "v": return 9
        case "x": return 7
        case "z": return 6
        case "w": return 13
        case "q": return 12
        default: return 36
        }
    }
}
