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
    let baseCtx = NSGraphicsContext(bitmapImageRep: rep)!
    let cg = baseCtx.cgContext
    cg.translateBy(x: 0, y: CGFloat(H))
    cg.scaleBy(x: 1, y: -1)
    let flippedCtx = NSGraphicsContext(cgContext: cg, flipped: true)
    NSGraphicsContext.current = flippedCtx
    draw(cg)
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
    drawDesktopChrome(ctx, title: "Notes", winX: 64, winY: 54, winW: 512, winH: 270)
    fillRect(ctx, 74, 61, 16, 16, 0.96, 0.30, 0.28, radius: 8)
    drawText("Meeting Notes - Q4 Roadmap", x: 96, y: 110, size: 16, bold: true, color: .black)
    drawText("- Ship Metal 4 Vision Engine", x: 96, y: 145, size: 13, bold: false, color: .darkGray)
}
tasks.append(TaskSpec(id: "03_close_window", image: "bench/images/03_close_window.png",
                      prompt: "Click the red close button in the top-left corner of the window.",
                      expectedAction: "click", targetBox1000: normBox(70, 58, 24, 24)))

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

// 11. macOS Top Menu Bar "View" item
_ = saveScene("11_menubar_view") { ctx in
    drawDesktopChrome(ctx, title: "Xcode", winX: 40, winY: 48, winW: 560, winH: 300, dark: true)
    let menus: [(String, CGFloat)] = [("Xcode", 18), ("File", 76), ("Edit", 118), ("View", 162), ("Navigate", 214), ("Window", 290), ("Help", 360)]
    for (m, mx) in menus {
        drawText(m, x: mx, y: 4, size: 13, bold: m == "Xcode", color: .white)
    }
}
tasks.append(TaskSpec(id: "11_menubar_view", image: "bench/images/11_menubar_view.png",
                      prompt: "Click 'View' in the top macOS menu bar.",
                      expectedAction: "click", targetBox1000: normBox(154, 0, 48, 24)))

// 12. macOS Bottom Dock "Terminal" Icon
_ = saveScene("12_dock_terminal") { ctx in
    drawDesktopChrome(ctx, title: "Desktop", winX: 70, winY: 36, winW: 500, winH: 250)
    // Dock container at bottom
    fillRect(ctx, 130, 316, 380, 58, 0.18, 0.22, 0.30, radius: 14)
    strokeRect(ctx, 130, 316, 380, 58, 0.40, 0.45, 0.55, radius: 14)
    let apps: [(String, CGFloat, CGFloat, CGFloat)] = [
        ("Finder", 0.20, 0.55, 0.95),
        ("Safari", 0.15, 0.68, 0.92),
        ("Terminal", 0.12, 0.14, 0.16),
        ("Mail", 0.22, 0.50, 0.90),
        ("Settings", 0.45, 0.48, 0.52),
    ]
    for (i, a) in apps.enumerated() {
        let ax = 148 + CGFloat(i) * 70
        fillRect(ctx, ax, 322, 52, 34, a.1, a.2, a.3, radius: 8)
        if a.0 == "Terminal" {
            drawText(">_", x: ax + 14, y: 330, size: 14, bold: true, color: .white)
        }
        drawText(a.0, x: ax + 2, y: 358, size: 10, bold: true, color: .white)
    }
}
tasks.append(TaskSpec(id: "12_dock_terminal", image: "bench/images/12_dock_terminal.png",
                      prompt: "Click the 'Terminal' icon in the bottom Dock.",
                      expectedAction: "click", targetBox1000: normBox(288, 320, 56, 52)))

// 13. Browser '+' New Tab Button
_ = saveScene("13_browser_new_tab") { ctx in
    drawDesktopChrome(ctx, title: "Browser", winX: 30, winY: 40, winW: 580, winH: 310)
    // Tab bar
    fillRect(ctx, 110, 44, 110, 24, 0.96, 0.96, 0.98, radius: 6)
    drawText("Docs", x: 148, y: 49, size: 12, bold: true, color: .black)
    fillRect(ctx, 226, 44, 110, 24, 0.82, 0.84, 0.88, radius: 6)
    drawText("GitHub", x: 260, y: 49, size: 12, bold: false, color: .darkGray)
    // '+' New Tab button
    fillRect(ctx, 344, 44, 28, 24, 0.78, 0.80, 0.85, radius: 6)
    drawText("+", x: 353, y: 46, size: 16, bold: true, color: .black)
}
tasks.append(TaskSpec(id: "13_browser_new_tab", image: "bench/images/13_browser_new_tab.png",
                      prompt: "Click the '+' button next to the 'GitHub' tab to open a new browser tab.",
                      expectedAction: "click", targetBox1000: normBox(344, 44, 28, 24)))

// 14. Preferences Checkbox "Remember window position"
_ = saveScene("14_checkbox_remember") { ctx in
    drawDesktopChrome(ctx, title: "Preferences", winX: 110, winY: 55, winW: 420, winH: 270)
    drawText("General Preferences", x: 140, y: 98, size: 16, bold: true, color: .black)
    let opts = ["Launch at macOS Login", "Remember window position", "Automatically install updates"]
    for (i, opt) in opts.enumerated() {
        let cy = 140 + CGFloat(i) * 44
        fillRect(ctx, 145, cy, 20, 20, 1.0, 1.0, 1.0, radius: 4)
        strokeRect(ctx, 145, cy, 20, 20, 0.45, 0.48, 0.54, radius: 4, lineWidth: 1.5)
        drawText(opt, x: 176, y: cy + 2, size: 13, bold: false, color: .black)
    }
}
tasks.append(TaskSpec(id: "14_checkbox_remember", image: "bench/images/14_checkbox_remember.png",
                      prompt: "Click the checkbox next to 'Remember window position'.",
                      expectedAction: "click", targetBox1000: normBox(145, 184, 200, 22)))

// 15. Radio Button "PNG Image"
_ = saveScene("15_radio_export_png") { ctx in
    drawDesktopChrome(ctx, title: "Export Format", winX: 130, winY: 60, winW: 380, winH: 260)
    drawText("Select Export Format:", x: 160, y: 105, size: 15, bold: true, color: .black)
    let formats = ["PDF Document", "PNG Image", "SVG Vector"]
    for (i, fmt) in formats.enumerated() {
        let ry = 145 + CGFloat(i) * 42
        fillRect(ctx, 165, ry, 20, 20, 1.0, 1.0, 1.0, radius: 10)
        strokeRect(ctx, 165, ry, 20, 20, 0.35, 0.40, 0.50, radius: 10, lineWidth: 1.8)
        drawText(fmt, x: 196, y: ry + 2, size: 13, bold: false, color: .black)
    }
}
tasks.append(TaskSpec(id: "15_radio_export_png", image: "bench/images/15_radio_export_png.png",
                      prompt: "Click the 'PNG Image' option.",
                      expectedAction: "click", targetBox1000: normBox(165, 187, 140, 22)))

// 16. Segmented Control "Week"
_ = saveScene("16_segmented_week") { ctx in
    drawDesktopChrome(ctx, title: "Calendar", winX: 50, winY: 45, winW: 540, winH: 295)
    fillRect(ctx, 190, 90, 260, 32, 0.85, 0.86, 0.90, radius: 7)
    let segs = ["Day", "Week", "Month", "Year"]
    for (i, s) in segs.enumerated() {
        let sx = 192 + CGFloat(i) * 64
        if s == "Day" {
            fillRect(ctx, sx, 92, 62, 28, 1.0, 1.0, 1.0, radius: 6)
        }
        drawText(s, x: sx + 14, y: 98, size: 13, bold: s == "Day", color: .black)
    }
}
tasks.append(TaskSpec(id: "16_segmented_week", image: "bench/images/16_segmented_week.png",
                      prompt: "Click 'Week' in the segmented control bar.",
                      expectedAction: "click", targetBox1000: normBox(256, 90, 64, 32)))

// 17. Slider Volume Knob
_ = saveScene("17_slider_volume_knob") { ctx in
    drawDesktopChrome(ctx, title: "Sound Settings", winX: 90, winY: 65, winW: 460, winH: 240, dark: true)
    drawText("Output Volume", x: 120, y: 125, size: 15, bold: true, color: .white)
    // Track
    fillRect(ctx, 120, 165, 380, 8, 0.30, 0.32, 0.36, radius: 4)
    fillRect(ctx, 120, 165, 260, 8, 0.10, 0.52, 0.98, radius: 4)
    // Circular white knob at x=380, y=169
    fillRect(ctx, 366, 155, 28, 28, 1.0, 1.0, 1.0, radius: 14)
    strokeRect(ctx, 366, 155, 28, 28, 0.65, 0.68, 0.72, radius: 14)
}
tasks.append(TaskSpec(id: "17_slider_volume_knob", image: "bench/images/17_slider_volume_knob.png",
                      prompt: "Click the white circular slider knob on the 'Output Volume' slider.",
                      expectedAction: "click", targetBox1000: normBox(366, 155, 28, 28)))

// 18. Resolution Dropdown Menu
_ = saveScene("18_dropdown_resolution") { ctx in
    drawDesktopChrome(ctx, title: "Displays", winX: 80, winY: 55, winW: 480, winH: 270)
    drawText("Display Resolution", x: 115, y: 125, size: 14, bold: true, color: .black)
    fillRect(ctx, 270, 118, 240, 32, 1.0, 1.0, 1.0, radius: 6)
    strokeRect(ctx, 270, 118, 240, 32, 0.65, 0.68, 0.74, radius: 6)
    drawText("2560 x 1440 (Default)", x: 284, y: 126, size: 13, bold: false, color: .black)
    fillRect(ctx, 482, 121, 24, 26, 0.12, 0.50, 0.96, radius: 5)
    drawText("v", x: 490, y: 126, size: 12, bold: true, color: .white)
}
tasks.append(TaskSpec(id: "18_dropdown_resolution", image: "bench/images/18_dropdown_resolution.png",
                      prompt: "Click the '2560 x 1440 (Default)' dropdown menu.",
                      expectedAction: "click", targetBox1000: normBox(270, 118, 240, 32)))

// 19. Table Row "Edit" Button for "charlie"
_ = saveScene("19_table_row_edit") { ctx in
    drawDesktopChrome(ctx, title: "Team Members", winX: 60, winY: 45, winW: 520, winH: 295)
    let users = ["alice@team.io", "bob@team.io", "charlie@team.io", "david@team.io"]
    for (i, u) in users.enumerated() {
        let ry = 95 + CGFloat(i) * 52
        fillRect(ctx, 85, ry, 470, 42, 1.0, 1.0, 1.0, radius: 6)
        strokeRect(ctx, 85, ry, 470, 42, 0.82, 0.84, 0.88, radius: 6)
        drawText(u, x: 102, y: ry + 13, size: 13, bold: false, color: .black)
        fillRect(ctx, 475, ry + 7, 66, 28, 0.12, 0.48, 0.94, radius: 6)
        drawText("Edit", x: 495, y: ry + 13, size: 12, bold: true, color: .white)
    }
}
tasks.append(TaskSpec(id: "19_table_row_edit", image: "bench/images/19_table_row_edit.png",
                      prompt: "Click the blue 'Edit' button in the row for 'charlie@team.io'.",
                      expectedAction: "click", targetBox1000: normBox(475, 206, 66, 28)))

// 20. Dark IDE Top Toolbar "Run" Button
_ = saveScene("20_ide_run_button") { ctx in
    drawDesktopChrome(ctx, title: "Code Studio", winX: 30, winY: 38, winW: 580, winH: 315, dark: true)
    // Toolbar buttons
    fillRect(ctx, 450, 42, 68, 24, 0.16, 0.72, 0.34, radius: 6)
    drawText("Run", x: 472, y: 46, size: 12, bold: true, color: .white)
    fillRect(ctx, 526, 42, 68, 24, 0.35, 0.36, 0.40, radius: 6)
    drawText("Stop", x: 546, y: 46, size: 12, bold: false, color: .white)
    fillRect(ctx, 45, 82, 550, 255, 0.11, 0.12, 0.15, radius: 6)
    drawText("func main() { print(\"Hello Metal 4\") }", x: 65, y: 105, size: 13, bold: false, color: .white)
}
tasks.append(TaskSpec(id: "20_ide_run_button", image: "bench/images/20_ide_run_button.png",
                      prompt: "Click the green 'Run' button in the top-right toolbar.",
                      expectedAction: "click", targetBox1000: normBox(450, 42, 68, 24)))

// 21. Floating Right-Click Context Menu "Duplicate"
_ = saveScene("21_context_menu_duplicate") { ctx in
    drawDesktopChrome(ctx, title: "Finder", winX: 50, winY: 45, winW: 540, winH: 295)
    // Context menu floating at (240, 100)
    fillRect(ctx, 240, 100, 170, 175, 0.98, 0.98, 1.0, radius: 8)
    strokeRect(ctx, 240, 100, 170, 175, 0.72, 0.74, 0.78, radius: 8)
    let items = ["Open", "Rename...", "Duplicate", "Compress", "Move to Trash"]
    for (i, item) in items.enumerated() {
        let iy = 108 + CGFloat(i) * 32
        if item == "Duplicate" {
            fillRect(ctx, 246, iy, 158, 28, 0.12, 0.48, 0.96, radius: 5)
        }
        drawText(item, x: 258, y: iy + 6, size: 13, bold: item == "Duplicate", color: item == "Duplicate" ? .white : .black)
    }
}
tasks.append(TaskSpec(id: "21_context_menu_duplicate", image: "bench/images/21_context_menu_duplicate.png",
                      prompt: "Click 'Duplicate' in the context menu.",
                      expectedAction: "click", targetBox1000: normBox(246, 172, 158, 28)))

// 22. Mail Client Toolbar "Compose" Button
_ = saveScene("22_mail_compose") { ctx in
    drawDesktopChrome(ctx, title: "Mail", winX: 40, winY: 40, winW: 560, winH: 310)
    let btns: [(String, CGFloat)] = [("Compose", 140), ("Reply", 235), ("Archive", 320), ("Delete", 415)]
    for (lbl, bx) in btns {
        fillRect(ctx, bx, 82, 82, 30, lbl == "Compose" ? 0.10 : 0.86, lbl == "Compose" ? 0.48 : 0.87, lbl == "Compose" ? 0.96 : 0.90, radius: 6)
        drawText(lbl, x: bx + 12, y: 89, size: 12, bold: lbl == "Compose", color: lbl == "Compose" ? .white : .black)
    }
}
tasks.append(TaskSpec(id: "22_mail_compose", image: "bench/images/22_mail_compose.png",
                      prompt: "Click the blue 'Compose' button in the toolbar.",
                      expectedAction: "click", targetBox1000: normBox(140, 82, 82, 30)))

// 23. Wi-Fi List: Click "Join" next to "Corp-Guest"
_ = saveScene("23_wifi_connect") { ctx in
    drawDesktopChrome(ctx, title: "Wi-Fi Networks", winX: 100, winY: 50, winW: 440, winH: 280)
    let nets = ["Home-5G-Ultra", "Corp-Guest", "CoffeeShop-Free"]
    for (i, net) in nets.enumerated() {
        let ny = 105 + CGFloat(i) * 58
        fillRect(ctx, 125, ny, 390, 44, 1.0, 1.0, 1.0, radius: 8)
        strokeRect(ctx, 125, ny, 390, 44, 0.80, 0.82, 0.86, radius: 8)
        drawText(net, x: 145, y: ny + 14, size: 14, bold: true, color: .black)
        fillRect(ctx, 430, ny + 8, 70, 28, 0.10, 0.48, 0.95, radius: 6)
        drawText("Join", x: 452, y: ny + 14, size: 12, bold: true, color: .white)
    }
}
tasks.append(TaskSpec(id: "23_wifi_connect", image: "bench/images/23_wifi_connect.png",
                      prompt: "Click the 'Join' button next to 'Corp-Guest'.",
                      expectedAction: "click", targetBox1000: normBox(430, 171, 70, 28)))

// 24. Bottom-Right Pagination "Next >" Button
_ = saveScene("24_pagination_next") { ctx in
    drawDesktopChrome(ctx, title: "Logs Viewer", winX: 45, winY: 40, winW: 550, winH: 310)
    let pages: [(String, CGFloat, CGFloat)] = [("1", 340, 36), ("2", 384, 36), ("3", 428, 36), ("Next >", 472, 74)]
    for (lbl, px, pw) in pages {
        fillRect(ctx, px, 298, pw, 32, lbl == "Next >" ? 0.12 : 0.86, lbl == "Next >" ? 0.48 : 0.88, lbl == "Next >" ? 0.95 : 0.90, radius: 6)
        drawText(lbl, x: px + 12, y: 306, size: 12, bold: lbl == "Next >", color: lbl == "Next >" ? .white : .black)
    }
}
tasks.append(TaskSpec(id: "24_pagination_next", image: "bench/images/24_pagination_next.png",
                      prompt: "Click the 'Next >' button at the bottom right.",
                      expectedAction: "click", targetBox1000: normBox(472, 298, 74, 32)))

// 25. Modal Top-Right 'X' Close Button
_ = saveScene("25_modal_close_x") { ctx in
    drawDesktopChrome(ctx, title: "Dashboard", winX: 40, winY: 40, winW: 560, winH: 310, dark: true)
    // Modal card
    fillRect(ctx, 140, 85, 360, 210, 0.24, 0.26, 0.30, radius: 12)
    strokeRect(ctx, 140, 85, 360, 210, 0.45, 0.48, 0.55, radius: 12)
    drawText("Subscription Details", x: 165, y: 105, size: 16, bold: true, color: .white)
    // Top-right X button on modal
    fillRect(ctx, 456, 97, 30, 30, 0.88, 0.26, 0.26, radius: 15)
    drawText("X", x: 466, y: 104, size: 14, bold: true, color: .white)
}
tasks.append(TaskSpec(id: "25_modal_close_x", image: "bench/images/25_modal_close_x.png",
                      prompt: "Click the red 'X' button in the top-right corner of the 'Subscription Details' modal.",
                      expectedAction: "click", targetBox1000: normBox(456, 97, 30, 30)))

// 26. Terminal Tab "ssh prod-01"
_ = saveScene("26_terminal_split_tab") { ctx in
    drawDesktopChrome(ctx, title: "Terminal", winX: 40, winY: 42, winW: 560, winH: 305, dark: true)
    // Tab 1
    fillRect(ctx, 40, 74, 280, 28, 0.14, 0.15, 0.18)
    drawText("local — zsh", x: 135, y: 80, size: 12, bold: false, color: .lightGray)
    // Tab 2
    fillRect(ctx, 320, 74, 280, 28, 0.24, 0.26, 0.32)
    drawText("ssh prod-01", x: 415, y: 80, size: 12, bold: true, color: .white)
    drawText("ethan@macbook ~ % ", x: 58, y: 125, size: 13, bold: false, color: .green)
}
tasks.append(TaskSpec(id: "26_terminal_split_tab", image: "bench/images/26_terminal_split_tab.png",
                      prompt: "Click the 'ssh prod-01' terminal tab.",
                      expectedAction: "click", targetBox1000: normBox(320, 74, 280, 28)))

// 27. Color Swatch Palette: Orange Swatch
_ = saveScene("27_color_swatch_orange") { ctx in
    drawDesktopChrome(ctx, title: "Color Picker", winX: 120, winY: 70, winW: 400, winH: 230)
    drawText("Choose Accent Color:", x: 150, y: 120, size: 15, bold: true, color: .black)
    let colors: [(CGFloat, CGFloat, CGFloat)] = [
        (0.92, 0.22, 0.22), // Red
        (0.98, 0.56, 0.10), // Orange
        (0.20, 0.78, 0.35), // Green
        (0.12, 0.48, 0.96), // Blue
        (0.68, 0.28, 0.88), // Purple
    ]
    for (i, c) in colors.enumerated() {
        let cx = 150 + CGFloat(i) * 68
        fillRect(ctx, cx, 165, 50, 50, c.0, c.1, c.2, radius: 10)
    }
}
tasks.append(TaskSpec(id: "27_color_swatch_orange", image: "bench/images/27_color_swatch_orange.png",
                      prompt: "Click the orange square color swatch (second from the left).",
                      expectedAction: "click", targetBox1000: normBox(218, 165, 50, 50)))

// 28. Yellow Minimize Traffic Light Button
_ = saveScene("28_minimize_window") { ctx in
    drawDesktopChrome(ctx, title: "Preview", winX: 70, winY: 55, winW: 500, winH: 270)
    // Enlarge traffic lights slightly for clear hit box
    fillRect(ctx, 82, 63, 14, 14, 0.95, 0.35, 0.32, radius: 7)
    fillRect(ctx, 104, 63, 14, 14, 0.98, 0.76, 0.18, radius: 7)
    fillRect(ctx, 126, 63, 14, 14, 0.28, 0.78, 0.35, radius: 7)
    drawText("architecture_diagram.pdf", x: 235, y: 63, size: 12, bold: true, color: .black)
}
tasks.append(TaskSpec(id: "28_minimize_window", image: "bench/images/28_minimize_window.png",
                      prompt: "Click the yellow minimize button in the top-left corner of the window.",
                      expectedAction: "click", targetBox1000: normBox(98, 58, 24, 24)))

// 29. Left Sidebar "Commit & Push" Button
_ = saveScene("29_git_commit_push") { ctx in
    drawDesktopChrome(ctx, title: "Source Control", winX: 35, winY: 40, winW: 570, winH: 310, dark: true)
    // Left git panel
    fillRect(ctx, 35, 72, 190, 278, 0.20, 0.21, 0.25)
    drawText("Changes (2 files)", x: 48, y: 86, size: 12, bold: true, color: .white)
    fillRect(ctx, 46, 115, 168, 60, 0.13, 0.14, 0.17, radius: 6)
    drawText("Optimize GEMM", x: 54, y: 125, size: 12, bold: false, color: .lightGray)
    fillRect(ctx, 46, 190, 168, 36, 0.10, 0.48, 0.96, radius: 7)
    drawText("Commit & Push", x: 76, y: 201, size: 13, bold: true, color: .white)
}
tasks.append(TaskSpec(id: "29_git_commit_push", image: "bench/images/29_git_commit_push.png",
                      prompt: "Click the blue 'Commit & Push' button in the left panel.",
                      expectedAction: "click", targetBox1000: normBox(46, 190, 168, 36)))

// 30. macOS Permission Dialog "Open System Settings"
_ = saveScene("30_permission_allow") { ctx in
    drawDesktopChrome(ctx, title: "Screen Recording Permission", winX: 115, winY: 75, winW: 410, winH: 230)
    drawText("'instantcua' would like to record this computer's screen.", x: 140, y: 120, size: 13, bold: true, color: .black)
    drawText("Grant access in Security & Privacy preferences.", x: 140, y: 148, size: 12, bold: false, color: .darkGray)
    fillRect(ctx, 145, 235, 130, 36, 0.85, 0.86, 0.88, radius: 7)
    drawText("Deny", x: 192, y: 246, size: 13, bold: false, color: .black)
    fillRect(ctx, 290, 235, 205, 36, 0.08, 0.48, 0.96, radius: 7)
    drawText("Open System Settings", x: 318, y: 246, size: 13, bold: true, color: .white)
}
tasks.append(TaskSpec(id: "30_permission_allow", image: "bench/images/30_permission_allow.png",
                      prompt: "Click the 'Open System Settings' button.",
                      expectedAction: "click", targetBox1000: normBox(290, 235, 205, 36)))

let enc = JSONEncoder()
enc.outputFormatting = [.prettyPrinted, .sortedKeys]
let jsonData = try! enc.encode(tasks)
try! jsonData.write(to: URL(fileURLWithPath: "bench/tasks.json"))
print("Generated \(tasks.count) GUI benchmark tasks in bench/tasks.json and bench/images/*.png")
