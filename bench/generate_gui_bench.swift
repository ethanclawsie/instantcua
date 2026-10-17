#!/usr/bin/env swift
import AppKit
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

let W = 640
let H = 384

struct TaskSpec: Codable {
    let id: String
    let image: String
    let prompt: String
    let expectedAction: String
    // Normalized [0, 1000] bounding box [x1, y1, x2, y2] for click tasks
    let targetBox1000: [Int]?
}

func makeContext() -> CGContext {
    let cs = CGColorSpaceCreateDeviceRGB()
    return CGContext(
        data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4,
        space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )!
}

// Flip coordinate system so (0,0) is top-left (matching screen coordinates)
func withTopLeft(_ ctx: CGContext, _ body: () -> Void) {
    ctx.saveGState()
    ctx.translateBy(x: 0, y: CGFloat(H))
    ctx.scaleBy(x: 1, y: -1)
    body()
    ctx.restoreGState()
}

func fillRect(_ ctx: CGContext, _ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat, _ r: CGFloat, _ g: CGFloat, _ b: CGFloat, radius: CGFloat = 0) {
    ctx.setFillColor(red: r, green: g, blue: b, alpha: 1.0)
    let rect = CGRect(x: x, y: y, width: w, height: h)
    if radius > 0 {
        let path = CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil)
        ctx.addPath(path)
        ctx.fillPath()
    } else {
        ctx.fill(rect)
    }
}

func strokeRect(_ ctx: CGContext, _ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat, _ r: CGFloat, _ g: CGFloat, _ b: CGFloat, radius: CGFloat = 0, lineWidth: CGFloat = 1.5) {
    ctx.setStrokeColor(red: r, green: g, blue: b, alpha: 1.0)
    ctx.setLineWidth(lineWidth)
    let rect = CGRect(x: x, y: y, width: w, height: h)
    if radius > 0 {
        let path = CGPath(roundedRect: rect, cornerWidth: radius, cornerHeight: radius, transform: nil)
        ctx.addPath(path)
        ctx.strokePath()
    } else {
        ctx.stroke(rect)
    }
}

func drawText(_ text: String, x: CGFloat, y: CGFloat, size: CGFloat = 14, bold: Bool = false, color: NSColor = .white) {
    let font = bold ? NSFont.boldSystemFont(ofSize: size) : NSFont.systemFont(ofSize: size)
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: color,
    ]
    let nsStr = NSString(string: text)
    nsStr.draw(at: NSPoint(x: x, y: y), withAttributes: attrs)
}

func drawDesktopChrome(_ ctx: CGContext, title: String, winX: CGFloat = 60, winY: CGFloat = 44, winW: CGFloat = 520, winH: CGFloat = 300, dark: Bool = false) {
    // Desktop wallpaper background
    fillRect(ctx, 0, 0, CGFloat(W), CGFloat(H), 0.12, 0.16, 0.24)
    // macOS menu bar
    fillRect(ctx, 0, 0, CGFloat(W), 24, 0.08, 0.10, 0.15)
    // Window background
    if dark {
        fillRect(ctx, winX, winY, winW, winH, 0.16, 0.17, 0.20, radius: 10)
        fillRect(ctx, winX, winY, winW, 32, 0.22, 0.23, 0.27, radius: 10)
    } else {
        fillRect(ctx, winX, winY, winW, winH, 0.96, 0.96, 0.98, radius: 10)
        fillRect(ctx, winX, winY, winW, 32, 0.88, 0.89, 0.92, radius: 10)
    }
    // Traffic lights: close (red), minimize (yellow), zoom (green)
    fillRect(ctx, winX + 12, winY + 10, 12, 12, 0.95, 0.35, 0.32, radius: 6)
    fillRect(ctx, winX + 30, winY + 10, 12, 12, 0.96, 0.74, 0.25, radius: 6)
    fillRect(ctx, winX + 48, winY + 10, 12, 12, 0.28, 0.78, 0.35, radius: 6)
}

func normBox(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) -> [Int] {
    let x1 = Int((x / CGFloat(W) * 1000.0).rounded())
    let y1 = Int((y / CGFloat(H) * 1000.0).rounded())
    let x2 = Int(((x + w) / CGFloat(W) * 1000.0).rounded())
    let y2 = Int(((y + h) / CGFloat(H) * 1000.0).rounded())
    return [max(0, x1 - 15), max(0, y1 - 15), min(1000, x2 + 15), min(1000, y2 + 15)]
}

func saveScene(_ id: String, draw: (CGContext) -> Void) -> String {
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: W, pixelsHigh: H,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: W * 4, bitsPerPixel: 32
    )!
    NSGraphicsContext.saveGraphicsState()
    let nsCtx = NSGraphicsContext(bitmapImageRep: rep)!
    NSGraphicsContext.current = nsCtx
    let cg = nsCtx.cgContext
    withTopLeft(cg) {
        draw(cg)
    }
    NSGraphicsContext.restoreGraphicsState()
    let path = "bench/images/\(id).png"
    let data = rep.representation(using: .png, properties: [:])!
    try! data.write(to: URL(fileURLWithPath: path))
    return path
}

var tasks: [TaskSpec] = []

// 1. Dialog "Save Changes" button
_ = saveScene("01_dialog_save") { ctx in
    drawDesktopChrome(ctx, title: "TextEdit", winX: 120, winY: 80, winW: 400, winH: 220)
    drawText("Do you want to save changes to Document.txt?", x: 155, y: 135, size: 14, bold: true, color: .black)
    drawText("Your changes will be lost if you don't save them.", x: 155, y: 162, size: 12, bold: false, color: .darkGray)
    // Cancel button
    fillRect(ctx, 250, 235, 95, 34, 0.85, 0.86, 0.88, radius: 7)
    drawText("Cancel", x: 274, y: 244, size: 13, bold: false, color: .black)
    // Save button (blue primary)
    fillRect(ctx, 365, 235, 120, 34, 0.05, 0.48, 0.98, radius: 7)
    drawText("Save Changes", x: 378, y: 244, size: 13, bold: true, color: .white)
}
tasks.append(TaskSpec(id: "01_dialog_save", image: "bench/images/01_dialog_save.png",
                      prompt: "Click the 'Save Changes' button.",
                      expectedAction: "click", targetBox1000: normBox(365, 235, 120, 34)))

// 2. Browser Search Bar
_ = saveScene("02_browser_search") { ctx in
    drawDesktopChrome(ctx, title: "Safari", winX: 30, winY: 36, winW: 580, winH: 320)
    // URL / Search bar
    fillRect(ctx, 160, 42, 320, 24, 1.0, 1.0, 1.0, radius: 6)
    strokeRect(ctx, 160, 42, 320, 24, 0.70, 0.72, 0.76, radius: 6)
    drawText("Search or enter website name", x: 205, y: 47, size: 12, bold: false, color: .gray)
    // Page center search input
    drawText("WebSearch", x: 265, y: 130, size: 24, bold: true, color: .black)
    fillRect(ctx, 170, 175, 300, 38, 1.0, 1.0, 1.0, radius: 19)
    strokeRect(ctx, 170, 175, 300, 38, 0.20, 0.50, 0.95, radius: 19, lineWidth: 2.0)
    drawText("Search the web...", x: 195, y: 186, size: 14, bold: false, color: .gray)
}
tasks.append(TaskSpec(id: "02_browser_search", image: "bench/images/02_browser_search.png",
                      prompt: "Click the 'Search the web...' input box in the center of the page.",
                      expectedAction: "click", targetBox1000: normBox(170, 175, 300, 38)))

// 3. Unlabeled Red Close Button (Traffic Light)
_ = saveScene("03_close_window") { ctx in
    drawDesktopChrome(ctx, title: "Notes", winX: 90, winY: 60, winW: 460, winH: 260)
    drawText("Meeting Notes - Q4 Roadmap", x: 120, y: 115, size: 16, bold: true, color: .black)
    drawText("- Ship Metal 4 Vision Engine", x: 120, y: 150, size: 13, bold: false, color: .darkGray)
}
tasks.append(TaskSpec(id: "03_close_window", image: "bench/images/03_close_window.png",
                      prompt: "Click the red close button in the top-left corner of the window.",
                      expectedAction: "click", targetBox1000: normBox(96, 64, 24, 24)))

// 4. Dark Mode Settings Toggle
_ = saveScene("04_settings_toggle") { ctx in
    drawDesktopChrome(ctx, title: "System Settings", winX: 70, winY: 50, winW: 500, winH: 290, dark: true)
    drawText("Appearance & Display", x: 100, y: 100, size: 17, bold: true, color: .white)
    fillRect(ctx, 100, 135, 440, 50, 0.22, 0.24, 0.28, radius: 8)
    drawText("Enable ProMotion High Refresh Rate", x: 118, y: 152, size: 14, bold: false, color: .white)
    // Toggle pill at right
    fillRect(ctx, 470, 147, 50, 26, 0.20, 0.78, 0.35, radius: 13)
    fillRect(ctx, 496, 149, 22, 22, 1.0, 1.0, 1.0, radius: 11)
}
tasks.append(TaskSpec(id: "04_settings_toggle", image: "bench/images/04_settings_toggle.png",
                      prompt: "Click the green toggle switch next to 'Enable ProMotion High Refresh Rate'.",
                      expectedAction: "click", targetBox1000: normBox(470, 147, 50, 26)))

// 5. Calculator Button '7'
_ = saveScene("05_calculator_seven") { ctx in
    drawDesktopChrome(ctx, title: "Calculator", winX: 200, winY: 40, winW: 240, winH: 310, dark: true)
    fillRect(ctx, 215, 78, 210, 48, 0.10, 0.11, 0.13, radius: 6)
    drawText("0", x: 400, y: 88, size: 24, bold: true, color: .white)
    let labels = [
        ["7", "8", "9", "×"],
        ["4", "5", "6", "-"],
        ["1", "2", "3", "+"],
    ]
    for (r, row) in labels.enumerated() {
        for (c, lbl) in row.enumerated() {
            let bx = 215 + CGFloat(c) * 54
            let by = 140 + CGFloat(r) * 52
            let isOp = c == 3
            fillRect(ctx, bx, by, 48, 44, isOp ? 0.95 : 0.32, isOp ? 0.60 : 0.34, isOp ? 0.12 : 0.38, radius: 22)
            drawText(lbl, x: bx + 18, y: by + 12, size: 16, bold: true, color: .white)
        }
    }
}
tasks.append(TaskSpec(id: "05_calculator_seven", image: "bench/images/05_calculator_seven.png",
                      prompt: "Click the '7' button on the calculator.",
                      expectedAction: "click", targetBox1000: normBox(215, 140, 48, 44)))

// 6. Canvas Media Player: Unlabeled Green Play Icon Button
_ = saveScene("06_canvas_play_icon") { ctx in
    drawDesktopChrome(ctx, title: "QuickTime Player", winX: 80, winY: 45, winW: 480, winH: 295, dark: true)
    // Video viewport
    fillRect(ctx, 95, 85, 450, 195, 0.06, 0.07, 0.09, radius: 8)
    // Big circular green play button in the center of the canvas (unlabeled graphic icon)
    let cx: CGFloat = 320, cy: CGFloat = 182
    fillRect(ctx, cx - 32, cy - 32, 64, 64, 0.18, 0.78, 0.38, radius: 32)
    ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
    ctx.move(to: CGPoint(x: cx - 10, y: cy - 16))
    ctx.addLine(to: CGPoint(x: cx - 10, y: cy + 16))
    ctx.addLine(to: CGPoint(x: cx + 18, y: cy))
    ctx.closePath()
    ctx.fillPath()
}
tasks.append(TaskSpec(id: "06_canvas_play_icon", image: "bench/images/06_canvas_play_icon.png",
                      prompt: "Click the green circular play button in the center of the video player.",
                      expectedAction: "click", targetBox1000: normBox(288, 150, 64, 64)))

// 7. Sidebar "Downloads" Tab
_ = saveScene("07_finder_downloads") { ctx in
    drawDesktopChrome(ctx, title: "Finder", winX: 50, winY: 45, winW: 540, winH: 295)
    // Left sidebar
    fillRect(ctx, 50, 77, 145, 263, 0.90, 0.91, 0.94)
    let items = ["Desktop", "Documents", "Downloads", "Applications"]
    for (i, item) in items.enumerated() {
        let iy = 95 + CGFloat(i) * 38
        if item == "Downloads" {
            fillRect(ctx, 58, iy, 128, 30, 0.80, 0.84, 0.92, radius: 6)
        }
        drawText(item, x: 72, y: iy + 7, size: 13, bold: item == "Downloads", color: .black)
    }
    drawText("Select a folder from the sidebar", x: 260, y: 185, size: 14, bold: false, color: .gray)
}
tasks.append(TaskSpec(id: "07_finder_downloads", image: "bench/images/07_finder_downloads.png",
                      prompt: "Click 'Downloads' in the left sidebar.",
                      expectedAction: "click", targetBox1000: normBox(58, 171, 128, 30)))

// 8. Login Form: Click "Sign In" Button
_ = saveScene("08_login_submit") { ctx in
    drawDesktopChrome(ctx, title: "Portal Login", winX: 140, winY: 45, winW: 360, winH: 295)
    drawText("Developer Sign In", x: 245, y: 90, size: 18, bold: true, color: .black)
    fillRect(ctx, 185, 130, 270, 34, 1.0, 1.0, 1.0, radius: 6)
    strokeRect(ctx, 185, 130, 270, 34, 0.75, 0.77, 0.80, radius: 6)
    drawText("ethan@example.com", x: 198, y: 139, size: 13, bold: false, color: .black)
    fillRect(ctx, 185, 180, 270, 34, 1.0, 1.0, 1.0, radius: 6)
    strokeRect(ctx, 185, 180, 270, 34, 0.75, 0.77, 0.80, radius: 6)
    drawText("••••••••••••", x: 198, y: 189, size: 13, bold: false, color: .black)
    // Sign In button
    fillRect(ctx, 185, 235, 270, 38, 0.10, 0.45, 0.95, radius: 8)
    drawText("Sign In", x: 295, y: 246, size: 14, bold: true, color: .white)
}
tasks.append(TaskSpec(id: "08_login_submit", image: "bench/images/08_login_submit.png",
                      prompt: "Click the blue 'Sign In' button.",
                      expectedAction: "click", targetBox1000: normBox(185, 235, 270, 38)))

// 9. Delete Confirmation: Red "Delete File" Button
_ = saveScene("09_delete_confirm") { ctx in
    drawDesktopChrome(ctx, title: "Confirm Delete", winX: 130, winY: 85, winW: 380, winH: 210)
    drawText("Delete 'old_backup.tar.gz' permanently?", x: 165, y: 130, size: 14, bold: true, color: .black)
    drawText("This item will be deleted immediately.", x: 165, y: 158, size: 12, bold: false, color: .darkGray)
    fillRect(ctx, 235, 225, 100, 34, 0.85, 0.86, 0.88, radius: 7)
    drawText("Keep File", x: 254, y: 234, size: 13, bold: false, color: .black)
    fillRect(ctx, 355, 225, 120, 34, 0.90, 0.22, 0.20, radius: 7)
    drawText("Delete File", x: 378, y: 234, size: 13, bold: true, color: .white)
}
tasks.append(TaskSpec(id: "09_delete_confirm", image: "bench/images/09_delete_confirm.png",
                      prompt: "Click the red 'Delete File' button.",
                      expectedAction: "click", targetBox1000: normBox(355, 225, 120, 34)))

// 10. Top-Right Notification Banner "Update Now"
_ = saveScene("10_notification_banner") { ctx in
    drawDesktopChrome(ctx, title: "Editor", winX: 40, winY: 60, winW: 420, winH: 270)
    // Floating notification in top-right
    fillRect(ctx, 390, 34, 238, 64, 0.98, 0.98, 0.99, radius: 12)
    strokeRect(ctx, 390, 34, 238, 64, 0.78, 0.80, 0.84, radius: 12)
    drawText("Software Update Available", x: 404, y: 44, size: 11, bold: true, color: .black)
    drawText("macOS 26.3.1 is ready.", x: 404, y: 62, size: 11, bold: false, color: .darkGray)
    fillRect(ctx, 540, 46, 78, 38, 0.08, 0.48, 0.96, radius: 7)
    drawText("Update", x: 556, y: 58, size: 12, bold: true, color: .white)
}
tasks.append(TaskSpec(id: "10_notification_banner", image: "bench/images/10_notification_banner.png",
                      prompt: "Click the 'Update' button in the top-right notification banner.",
                      expectedAction: "click", targetBox1000: normBox(540, 46, 78, 38)))

let enc = JSONEncoder()
enc.outputFormatting = [.prettyPrinted, .sortedKeys]
let jsonData = try! enc.encode(tasks)
try! jsonData.write(to: URL(fileURLWithPath: "bench/tasks.json"))
print("Generated \(tasks.count) GUI benchmark tasks in bench/tasks.json and bench/images/*.png")
