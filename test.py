# app.py — Facial Paralysis Simulator (Final Production Build)
# Uses printlayout.jpg background and 3×2.3 capture frame ratio.

import os
import io
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
        [[int(p.x * w + 0.5), int(p.y * h + 0.5)] for p in lm], dtype=np.int32
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
        src_crop, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
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
    mask = convex_mask_from_indices(img_bgr.shape[:2], pts, region, feather_px=15)
    result = piecewise_warp(img_bgr, pts, dst_pts, tris, mask)
    result = cv2.bilateralFilter(result, d=4, sigmaColor=40, sigmaSpace=40)
    return result

# ---------------- Report Generation ----------------
def generate_report(images):
    if images is None:
        return None
    original_img, simulated_img = images
    
    try:
        script_dir = Path(__file__).parent
        bg_path = script_dir / "printlayout.jpg"
        bg_img = cv2.imread(str(bg_path))
    except NameError:
        bg_img = cv2.imread("printlayout.jpg")

    if bg_img is None:
        print("⚠️ Could not load printlayout.jpg.")
        return None
    bg_h, bg_w, _ = bg_img.shape

    # Resize faces to fit inside white boxes (3x2.3 ratio)
    target_h = int(bg_h * 0.48)
    target_w = int(target_h * (2.3 / 3.0))

    def prepare_face(img):
        # This function resizes the full image and ensures it's 3-channel BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.resize(img_bgr, (target_w, target_h))

    orig = prepare_face(original_img)
    sim = prepare_face(simulated_img)

    x_left = int(bg_w * 0.17)
    x_right = int(bg_w * 0.56)
    y_top = int(bg_h * 0.25)

    # Paste the resized full images directly
    bg_img[y_top : y_top + target_h, x_left : x_left + target_w] = orig
    bg_img[y_top : y_top + target_h, x_right : x_right + target_w] = sim

    return cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)


# ---------------- Gradio Interface ----------------
def build_interface():
    print_css = """
    @media print {
      body > * { visibility: hidden; }
      #popup-view, #popup-view * { visibility: visible; }
      #popup-view { position: absolute; left: 0; top: 0; width: 100%; }
      @page { size: auto; margin: 0; }
    }
    """
    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), css=print_css) as app:
        image_state = gr.State(None)

        with gr.Column(elem_id="main-view") as main_view:
            gr.Markdown(
                "# Automatic Facial Paralysis Simulator\n"
                "Look into the camera and click **Generate** to simulate facial paralysis effects."
            )
            status_display = gr.Markdown("")
            with gr.Row(equal_height=True):
                input_display = gr.Image(
                    sources=["webcam"],
                    type="numpy",
                    label="Input (3×2.3 frame)",
                    height=520,
                    width=400,
                    mirror_webcam=True,
                )
                output_display = gr.Image(
                    type="numpy",
                    label="Simulated Output (Aligned to print layout)",
                    height=520,
                    width=400,
                )
            with gr.Row():
                capture_button = gr.Button("Generate")
                report_button = gr.Button("Take a look", visible=False)

        with gr.Column(visible=False, elem_id="popup-view") as popup_view:
            gr.Markdown("## Report Preview")
            report_image_display = gr.Image(label="Final Report", interactive=False)
            with gr.Row():
                print_button_popup = gr.Button("Print")
                close_button_popup = gr.Button("Close")

        def capture_and_simulate(frame):
            if frame is None:
                return (gr.update(), gr.update(), gr.update(visible=False), None, "Please wait for webcam.")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            simulated_bgr = simulate(frame_bgr, side="right", severity=0.62, lateral=0.06)
            if simulated_bgr is None or np.array_equal(frame_bgr, simulated_bgr):
                return (frame, None, gr.update(visible=False), None, "⚠️ No face detected. Try again.")
            simulated_rgb = cv2.cvtColor(simulated_bgr, cv2.COLOR_BGR2RGB)
            return (frame, simulated_rgb, gr.update(visible=True), (frame, simulated_rgb), "✅ Simulation complete!")

        def show_popup(images):
            report_img = generate_report(images)
            return gr.update(visible=False), gr.update(visible=True), report_img

        def close_popup():
            return gr.update(visible=True), gr.update(visible=False), None

        capture_button.click(
            fn=capture_and_simulate,
            inputs=[input_display],
            outputs=[input_display, output_display, report_button, image_state, status_display],
        )
        report_button.click(
            fn=show_popup,
            inputs=[image_state],
            outputs=[main_view, popup_view, report_image_display],
        )
        close_button_popup.click(
            fn=close_popup,
            inputs=None,
            outputs=[main_view, popup_view, report_image_display],
        )
        print_button_popup.click(fn=None, js="() => { window.print(); }")
       

    return app


if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="localhost", server_port=7865, share=True)
