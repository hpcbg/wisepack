"""WISEPACK perception providers — the detector-aware edge of the system.

One module per perception method. A provider turns whatever its detector
produces into the domain-neutral `ObservationBatch` defined in
`wisepack_core.perception`, and that is the entire contract: nothing above this
package knows which provider ran, what it detects, or where it came from.

    fasterrcnn_bottle   Faster R-CNN + ArUco homography on a calibrated plane.
                        The implementation developed in HARMONY, reused.

Future siblings — YOLO/OBB, RGB-D pose estimation, segmentation — are added
here and selected with `WISEPACK_PERCEPTION_DETECTOR`. Nothing else moves.

Deliberately EMPTY of imports: loading this package must not drag torch, cv2 or
a camera into a process that only wants to name a provider.
"""
