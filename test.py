# app.py — Facial Paralysis Simulator (Final Production Build — aspect-safe)
# Uses printlayout.jpg background and 3×2.3 capture frame ratio.
# Preserves all your simulation logic and fixes vertical-stretch.

import os
import base64
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
import gradio as gr

# ---------------- Background image ----------------
BG_PATH = Path("./printlayout.jpg")
if not BG_PATH.exists():
    print("⚠️ Warning: printlayout.jpg not found in current directory.")
BG_DATA_URL = None
try:
    b64 = base64.b64encode(BG_PATH.read_bytes()).decode("ascii")
    BG_DATA_URL = f"data:image/jpeg;base64,{b64}"
except Exception:
    BG_DATA_URL = None

# ---------------- Landmark groups ----------------
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LIPS_REGION = sorted(set(LIPS_OUTER + LIPS_INNER))
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65]
RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295]
LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER = 61, 291
STABLE_ANCHORS = [33, 263, 61, 291]
CHIN_REGION = [
    152,
    377,
    400,
    378,
    379,
    365,
    397,
    288,
    361,
    172,
    58,
    132,
    93,
    168,
    417,
    200,
    428,
    199,
    175,
    152,
]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


# ---------------- Landmark extraction ----------------
def landmarks_from_image_rgb(img_rgb):
    h, w = img_rgb.shape[:2]
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array(
        [[int(p.x * w + 0.5), int(p.y * h + 0.5)] for p in lm],
        dtype=np.int32,
    )


# ---------------- Masks ----------------
def convex_mask_from_indices(shape, points, indices, feather_px=25):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if not indices:
        return mask
    pts = points[indices].astype(np.int32)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull, 255)
    k = max(1, (feather_px // 2) * 2 + 1)
    return cv2.GaussianBlur(mask, (k, k), 0)


# ---------------- Triangulation + warping ----------------
def triangulate_region(points, region_indices):
    if len(region_indices) < 3:
        return np.array([], dtype=np.int32)
    region_pts = points[region_indices].astype(np.float64)
    try:
        tri = Delaunay(region_pts)
    except Exception:
        return np.array([], dtype=np.int32)
    return np.array(
        [
            [region_indices[int(a)], region_indices[int(b)], region_indices[int(c)]]
            for a, b, c in tri.simplices
        ],
        dtype=np.int32,
    )


def warp_triangle(src_img, src_tri, dst_tri):
    x, y, w, h = cv2.boundingRect(np.array(dst_tri, dtype=np.int32))
    if w <= 0 or h <= 0:
        return None, None, (x, y, w, h)
    sx, sy, sw, sh = cv2.boundingRect(np.array(src_tri, dtype=np.int32))
    src_crop = src_img[sy : sy + sh, sx : sx + sw]
    if src_crop.size == 0:
        return None, None, (x, y, w, h)
    src_shift = np.float32([[src_tri[i][0] - sx, src_tri[i][1] - sy] for i in range(3)])
    dst_shift = np.float32([[dst_tri[i][0] - x, dst_tri[i][1] - y] for i in range(3)])
    M = cv2.getAffineTransform(src_shift, dst_shift)
    warped = cv2.warpAffine(
        src_crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_shift), 255)
    return warped, mask, (x, y, w, h)


def piecewise_warp(src_img, src_pts, dst_pts, triangles, region_mask):
    h, w = src_img.shape[:2]
    accum = np.zeros((h, w, 3), dtype=np.float32)
    counts = np.zeros((h, w, 1), dtype=np.float32)
    for tri in triangles:
        wp, mk, (x, y, ww, hh) = warp_triangle(src_img, src_pts[tri], dst_pts[tri])
        if wp is None:
            continue
        m = mk[:, :, None].astype(np.float32) / 255.0
        accum[y : y + hh, x : x + ww] += wp.astype(np.float32) * m
        counts[y : y + hh, x : x + ww] += m[:, :, :1]
    tri_mask = counts > 0
    counts[counts == 0] = 1.0
    averaged = accum / counts
    final_warped = src_img.astype(np.float32)
    np.copyto(final_warped, averaged, where=tri_mask)
    alpha = (region_mask.astype(np.float32) / 255.0)[:, :, None]
    blended = final_warped * alpha + src_img.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------- Geometry deformation ----------------
def compute_droop(pts, side="left", severity=0.58, lateral=0.05):
    dst = pts.astype(np.float32).copy()
    cx = np.mean(pts[:, 0])
    face_h = np.max(pts[:, 1]) - np.min(pts[:, 1])
    base = severity * (face_h * 0.18)
    lateral_px = lateral * (face_h * 0.08)
    if side.lower().startswith("l"):
        sel = pts[:, 0] < cx
        sign = 1.0
        eyes, brows, mouth_corner = LEFT_EYE, LEFT_BROW, LEFT_MOUTH_CORNER
    else:
        sel = pts[:, 0] > cx
        sign = -1.0
        eyes, brows, mouth_corner = RIGHT_EYE, RIGHT_BROW, RIGHT_MOUTH_CORNER
    lips = np.array(LIPS_REGION)
    mcx = np.mean(pts[lips, 0])
    maxdx = np.max(np.abs(pts[lips, 0] - mcx)) + 1e-6
    for i in lips:
        if sel[i]:
            dx = abs(pts[i, 0] - mcx)
            w = 0.25 + 0.75 * (dx / maxdx)
            dst[i, 1] += base * (0.8 + 0.2 * w) * w
            dst[i, 0] += sign * lateral_px * w
    if sel[mouth_corner]:
        dst[mouth_corner, 1] += base * 1.2
        dst[mouth_corner, 0] += sign * lateral_px * 0.9
    ecx = np.mean(pts[eyes, 0])
    ecy = np.mean(pts[eyes, 1])
    span = max(1e-6, np.max(pts[eyes, 0]) - np.min(pts[eyes, 0]))
    for i in eyes:
        if not sel[i]:
            continue
        lateralness = (
            ((pts[i, 0] - np.min(pts[eyes, 0])) / span)
            if sign > 0
            else ((np.max(pts[eyes, 0]) - pts[i, 0]) / span)
        )
        lateralness = float(np.clip(lateralness, 0.0, 1.0))
        center_w = 0.4 + 0.6 * (1 - abs(pts[i, 0] - ecx) / span)
        w = 0.45 * center_w + 0.55 * lateralness
        if pts[i, 1] < ecy:
            dst[i, 1] += base * 0.40 * w
        else:
            dst[i, 1] -= base * 0.14 * w
    mid_x = cx
    max_side = max(1e-6, np.max(np.abs(pts[brows, 0] - mid_x)))
    for i in brows:
        if sel[i]:
            side_w = np.clip(abs(pts[i, 0] - mid_x) / max_side, 0.0, 1.0)
            bw = 0.18 + 0.82 * side_w
            dst[i, 1] += base * 0.24 * bw
            dst[i, 0] += sign * lateral_px * 0.07 * bw
    for i in CHIN_REGION:
        if sel[i]:
            dst[i, 1] += base * 0.18
            dst[i, 0] += sign * lateral_px * 0.11
    return dst


# ---------------- Main simulation ----------------
def simulate(img_bgr, side="left", severity=0.58, lateral=0.05):
    if img_bgr is None:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pts = landmarks_from_image_rgb(rgb)
    if pts is None:
        return img_bgr
    dst_pts = compute_droop(pts, side, severity, lateral)
    region = sorted(
        set(
            LIPS_REGION
            + LEFT_EYE
            + RIGHT_EYE
            + LEFT_BROW
            + RIGHT_BROW
            + STABLE_ANCHORS
            + CHIN_REGION
        )
    )
    tris = triangulate_region(pts, region)
    if tris.size == 0:
        return img_bgr
    mask = convex_mask_from_indices(img_bgr.shape[:2], pts, region, feather_px=35)
    result = piecewise_warp(img_bgr, pts, dst_pts, tris, mask)
    result = cv2.bilateralFilter(result, d=4, sigmaColor=40, sigmaSpace=40)
    return result


# ---------------- Report Generation (ASPECT-SAFE) ----------------
def generate_report(images):
    if not images or images[0] is None or images[1] is None:
        return None

    original_img, simulated_img = images
    bg_path = Path("printlayout.jpg")

    # Load background
    bg_img = cv2.imread(str(bg_path)) if bg_path.exists() else None
    if bg_img is None:
        bg_img = np.ones((1500, 2100, 3), dtype=np.uint8) * 255

    # --- FINAL portrait placement (based on your layout) ---
    frame_boxes = {
        "left":  {"x": 300,  "y": 325, "w": 690, "h": 890},
        "right": {"x": 1115, "y": 325, "w": 690, "h": 890},
    }

    def fit_portrait_crop(img_rgb, box, scale=0.93):
        """Crop image to portrait and scale down slightly to expose border."""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ih, iw = img_bgr.shape[:2]
        aspect_img = iw / ih
        aspect_box = box["w"] / box["h"]

        # Step 1: crop horizontally if needed
        if aspect_img > aspect_box:
            new_w = int(ih * aspect_box)
            x0 = (iw - new_w) // 2
            img_bgr = img_bgr[:, x0:x0 + new_w]
        else:
            # pad vertically (black)
            pad = int(((iw / aspect_box) - ih) / 2)
            if pad > 0:
                img_bgr = cv2.copyMakeBorder(
                    img_bgr, pad, pad, 0, 0,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

        # Step 2: scale image a bit smaller (no white border)
        new_w = int(box["w"] * scale)
        new_h = int(box["h"] * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Step 3: compute offsets for centering inside colored frame
        x_offset = box["x"] + (box["w"] - new_w) // 2
        y_offset = box["y"] + (box["h"] - new_h) // 2

        # Step 4: overlay the shrunken image directly on top of the layout
        bg_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Apply both
    fit_portrait_crop(original_img, frame_boxes["left"], scale=0.93)
    fit_portrait_crop(simulated_img, frame_boxes["right"], scale=0.93)

    return cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)



# ---------------- Gradio UI ----------------
def build_interface():
    css = """
    footer, header, [data-testid*='toolbar'], [class*='image-toolbar'], .icon-button {
        display: none !important;
    }
    .gradio-container { background: #fff !important; }
    @media print {
      body * { visibility: hidden !important; }
      #print-area, #print-area * { visibility: visible !important; }
      #print-area { position: fixed; left: 0; top: 0; width: 100vw; height: 100vh; background: white; }
      #print-area img { width: 100vw; height: 100vh; object-fit: contain; }
      @page { size: A4 landscape; margin: 0; }
      #print-btn, #reset-btn { display: none !important; }
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Default(primary_hue="blue")) as app:
        image_state = gr.State(None)

        with gr.Column(elem_id="main-view") as main_view:
            gr.Markdown("# Facial Paralysis Simulator")
            status_display = gr.Markdown("")
            with gr.Row():
                input_display = gr.Image(
                    sources=["webcam"], type="numpy", mirror_webcam=True, label=None
                )
                output_display = gr.Image(type="numpy", label=None)
            capture_button = gr.Button("Generate")

        with gr.Column(visible=False, elem_id="popup-view") as popup_view:
            report_image_display = gr.Image(
                label=None,
                interactive=False,
                elem_id="print-area",
                show_download_button=False,
                show_label=False,
                show_share_button=False,
            )
            with gr.Row():
                print_button = gr.Button("🖨️ Print", elem_id="print-btn")
                reset_button = gr.Button("↺ Reset", elem_id="reset-btn")

        def capture_and_simulate(frame):
            if frame is None:
                return (gr.update(), gr.update(), None, "Please wait for webcam.")
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            sim_bgr = simulate(bgr, side="right", severity=0.62, lateral=0.06)
            if sim_bgr is None or np.array_equal(sim_bgr, bgr):
                return (frame, None, None, "⚠️ No face detected.")
            sim_rgb = cv2.cvtColor(sim_bgr, cv2.COLOR_BGR2RGB)
            return (frame, sim_rgb, (frame, sim_rgb), "✅ Simulation complete!")

        def show_popup(images):
            # images is either None or (orig_rgb, sim_rgb)
            report_img = generate_report(images)
            if report_img is None:
                # show main view again (nothing to preview)
                return gr.update(visible=True), gr.update(visible=False), None
            # return final RGB image to Gradio Image (works)
            return gr.update(visible=False), gr.update(visible=True), report_img

        def reset_app():
            return gr.update(visible=True), gr.update(visible=False), None

        capture_button.click(
            fn=capture_and_simulate,
            inputs=[input_display],
            outputs=[input_display, output_display, image_state, status_display],
        ).then(
            fn=show_popup,
            inputs=[image_state],
            outputs=[main_view, popup_view, report_image_display],
        )

        print_button.click(fn=None, js="() => { window.print(); }")
        reset_button.click(
            fn=reset_app, outputs=[main_view, popup_view, report_image_display]
        )

    return app


if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="localhost", server_port=7865)
