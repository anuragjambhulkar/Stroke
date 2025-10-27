# app.py — Facial Paralysis Simulator (Natural & Realistic)

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
import gradio as gr

# ---------------- Landmark groups ----------------
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291]
LIPS_INNER = [78,95,88,178,87,14,317,402,318,324,308]
LIPS_REGION = sorted(set(LIPS_OUTER + LIPS_INNER))
LEFT_EYE  = [33,7,163,144,145,153,154,155,133]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263]
LEFT_BROW  = [70,63,105,66,107,55,65]
RIGHT_BROW = [336,296,334,293,300,285,295]
LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER = 61, 291
STABLE_ANCHORS = [33,263,61,291, 172, 148, 152, 377, 397]


mp_face_mesh = mp.solutions.face_mesh

# ---------------- Landmarks ----------------
def landmarks_from_image_rgb(img_rgb):
    h, w = img_rgb.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                               refine_landmarks=True, min_detection_confidence=0.5) as fm:
        res = fm.process(img_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[int(p.x*w + 0.5), int(p.y*h + 0.5)] for p in lm], dtype=np.int32)

# ---------------- Masks ----------------
def convex_mask_from_indices(shape, points, indices, feather_px=25):  # Softer feathering
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if not indices:
        return mask
    pts = points[indices].astype(np.int32)
    hull = cv2.convexHull(pts.reshape(-1,1,2))
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
    return np.array([[region_indices[int(a)], region_indices[int(b)], region_indices[int(c)]]
                     for a,b,c in tri.simplices], dtype=np.int32)

def warp_triangle(src_img, src_tri, dst_tri):
    x, y, w, h = cv2.boundingRect(np.array(dst_tri, dtype=np.int32))
    if w <= 0 or h <= 0:
        return None, None, (x,y,w,h)
    sx, sy, sw, sh = cv2.boundingRect(np.array(src_tri, dtype=np.int32))
    src_crop = src_img[sy:sy+sh, sx:sx+sw]
    if src_crop.size == 0:
        return None, None, (x,y,w,h)
    src_shift = np.float32([[src_tri[i][0]-sx, src_tri[i][1]-sy] for i in range(3)])
    dst_shift = np.float32([[dst_tri[i][0]-x,  dst_tri[i][1]-y ] for i in range(3)])
    M = cv2.getAffineTransform(src_shift, dst_shift)
    warped = cv2.warpAffine(src_crop, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_shift), 255)
    return warped, mask, (x,y,w,h)

def piecewise_warp(src_img, src_pts, dst_pts, triangles, region_mask):
    h, w = src_img.shape[:2]
    accum = np.zeros((h,w,3), dtype=np.float32)
    acc_mask = np.zeros((h,w), dtype=np.uint8)
    for tri in triangles:
        wp, mk, (x,y,ww,hh) = warp_triangle(src_img, src_pts[tri], dst_pts[tri])
        if wp is None:
            continue
        m = (mk[:,:,None].astype(np.float32)/255.0)
        accum[y:y+hh, x:x+ww] += wp.astype(np.float32) * m
        acc_mask[y:y+hh, x:x+ww] = np.maximum(acc_mask[y:y+hh, x:x+ww], mk)
    final = src_img.copy().astype(np.float32)
    area = acc_mask > 0
    final[np.dstack([area]*3)] = accum[np.dstack([area]*3)]
    alpha = (cv2.GaussianBlur(region_mask, (21,21), 0).astype(np.float32)/255.0)[:,:,None]  # wider blur for smoother blending
    return np.clip(final * alpha + src_img.astype(np.float32) * (1.0 - alpha),0,255).astype(np.uint8)

# ---------------- Geometry ----------------
def compute_droop(pts, side='left', severity=0.58, lateral=0.05):
    dst = pts.astype(np.float32).copy()
    cx = np.mean(pts[:,0])
    face_h = np.max(pts[:,1]) - np.min(pts[:,1])
    base = severity * (face_h * 0.18)
    lateral_px = lateral * (face_h * 0.08)

    if side.lower().startswith('l'):
        sel = pts[:,0] < cx; sign = 1.0
        eyes, brows, mouth_corner = LEFT_EYE, LEFT_BROW, LEFT_MOUTH_CORNER
    else:
        sel = pts[:,0] > cx; sign = -1.0
        eyes, brows, mouth_corner = RIGHT_EYE, RIGHT_BROW, RIGHT_MOUTH_CORNER

    lips = np.array(LIPS_REGION)
    mcx = np.mean(pts[lips,0]); maxdx = np.max(np.abs(pts[lips,0]-mcx)) + 1e-6
    for i in lips:
        if sel[i]:
            dx = abs(pts[i,0]-mcx); w = 0.25 + 0.75*(dx/maxdx)
            dst[i,1] += base * (0.8 + 0.2*w) * w
            dst[i,0] += sign * lateral_px * w
    if sel[mouth_corner]:
        dst[mouth_corner,1] += base * 1.2
        dst[mouth_corner,0] += sign * lateral_px * 0.9

    ecx = np.mean(pts[eyes,0]); ecy = np.mean(pts[eyes,1])
    span = max(1e-6, np.max(pts[eyes,0]) - np.min(pts[eyes,0]))
    for i in eyes:
        if not sel[i]:
            continue
        lateralness = ((pts[i,0] - np.min(pts[eyes,0])) / span) if sign > 0 else ((np.max(pts[eyes,0]) - pts[i,0]) / span)
        lateralness = float(np.clip(lateralness, 0.0, 1.0))
        center_w = 0.4 + 0.6*(1 - abs(pts[i,0]-ecx)/span)
        w = 0.45*center_w + 0.55*lateralness
        if pts[i,1] < ecy:
            dst[i,1] += base * 0.40 * w
        else:
            dst[i,1] -= base * 0.14 * w

    mid_x = cx
    max_side = max(1e-6, np.max(np.abs(pts[brows,0] - mid_x)))
    for i in brows:
        if sel[i]:
            side_w = np.clip(abs(pts[i,0] - mid_x)/max_side, 0.0, 1.0)
            bw = 0.18 + 0.82*side_w
            dst[i,1] += base * 0.24 * bw
            dst[i,0] += sign * lateral_px * 0.07 * bw


    return dst

# ---------------- Main processing ----------------
def simulate(img_bgr, side='left', severity=0.58, lateral=0.05):
    if img_bgr is None:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pts = landmarks_from_image_rgb(rgb)
    if pts is None:
        return img_bgr
    dst_pts = compute_droop(pts, side, severity, lateral)
    region = sorted(set(LIPS_REGION + LEFT_EYE + RIGHT_EYE + LEFT_BROW + RIGHT_BROW + STABLE_ANCHORS))
    tris = triangulate_region(pts, region)
    if tris.size == 0:
        return img_bgr
    mask = convex_mask_from_indices(img_bgr.shape[:2], pts, region, feather_px=25)
    result = piecewise_warp(img_bgr, pts, dst_pts, tris, mask)

    # === Natural skin texture pass ===
    # Apply a gentle blur to smooth transitions and reduce harshness.
    # A bilateral filter preserves edges while smoothing, creating a more natural look.
    # The 'd' parameter is the diameter of the pixel neighborhood. A small value keeps it localized.
    # sigmaColor and sigmaSpace control the intensity of the blur.
    result = cv2.bilateralFilter(result, d=4, sigmaColor=40, sigmaSpace=40)

    return result

# ---------------- Gradio UI ----------------
def build_interface():
    return gr.Interface(
        fn=simulate,
        inputs=[
            gr.Image(type="numpy"),
            gr.Radio(['left','right'], value='left', label='Side'),
            gr.Slider(0.0, 1.0, value=0.62, label='Droop Intensity'),
            gr.Slider(-0.5, 0.5, value=0.06, label='Mouth Stretch'),
        ],
        outputs=gr.Image(type="numpy"),
        title="Facial Paralysis Simulator — Natural & Realistic",
        description="Simulates facial paralysis with a focus on natural-looking skin texture and subtle blending."
    )

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7866, share=True)