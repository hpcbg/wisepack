"""Project the ORIGINAL Cylinder5 CAD under a pose, versus the exact mask.

Runs inside the FoundationPose worker image, which already has trimesh and cv2.
Deliberately uses the STL AS IT IS ON DISK: this is the check that catches a
frame or origin convention mistake, so it must not share any code with the
transform being checked.
"""
import json, sys
import numpy as np, trimesh, cv2

T = np.array(json.loads(sys.argv[1]), dtype=np.float64)
tag = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else ""

mask = cv2.imread("/frame/masks/000000.png", cv2.IMREAD_UNCHANGED) > 0
K = np.loadtxt("/frame/cam_K.txt").reshape(3, 3)
H, W = mask.shape
mesh = trimesh.load("/ref/CAD-Models/STL-Files/Cylinder5.stl")
V = np.asarray(mesh.vertices, dtype=np.float64) / 1000.0
F = np.asarray(mesh.faces, dtype=np.int32)

P = (T[:3, :3] @ V.T).T + T[:3, 3]
z = P[:, 2]
u = K[0, 0] * P[:, 0] / z + K[0, 2]
v = K[1, 1] * P[:, 1] / z + K[1, 2]
pts = np.stack([u, v], axis=1)
img = np.zeros((H, W), np.uint8)
tri = pts[F].astype(np.int32)
cv2.fillPoly(img, tri[(z[F] > 1e-6).all(axis=1)], 255)
silhouette = img > 0

inter = np.logical_and(silhouette, mask).sum()
union = np.logical_or(silhouette, mask).sum()
iou = inter / union if union else 0.0
ys, xs = np.nonzero(mask)
if silhouette.any():
    sy, sx = np.nonzero(silhouette)
    centroid = float(np.hypot(sx.mean() - xs.mean(), sy.mean() - ys.mean()))
else:
    centroid = float("nan")
print(f"REPROJECT {iou:.6f} {centroid:.4f}")

if out:
    rgb = cv2.imread("/frame/rgb/000000.png")
    overlay = rgb.copy()
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cs, -1, (255, 255, 255), 1)   # exact mask, white
    cs, _ = cv2.findContours(silhouette.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cs, -1, (0, 255, 0), 2)       # projected CAD, green
    cv2.imwrite(f"{out}/overlay_{tag}.png", overlay)
