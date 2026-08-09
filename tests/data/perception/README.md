# Perception test data

## `calibrated-scene.jpg`

A 1600 × 1200 camera frame of the printed ArUco calibration board with three
bottles on it — two with a visible cap, one without — used by
`tests/test_perception_detector_regression.py`.

**Why it is committed.** It is the reference frame for the detector regression:
the ported `fasterrcnn_bottle` provider must reproduce, from this exact frame and
the same weights, the measurement the pre-port implementation produced —

| object | x (mm) | y (mm) | yaw (deg) | confidence |
|---|---|---|---|---|
| 1 (selected) | 86.782 | 83.553 | −115.112 | 0.999 |
| 2 | 41.008 | 59.853 | 24.302 | 0.997 |

Committing 93 KB of JPEG is what lets that check run with **no camera, no
network and no other repository present** — which is the whole point of the
port. Regenerating it is not possible after the fact: it is a photograph of a
physical scene, and a new photograph would need new reference numbers.

**Provenance.** Captured in the HARMONY project
(`ai-bottle-detector-fiware/utils/images/`, MIT, "Copyright (c) 2026 Kaloyan
Yovchev") with the capture utility that ships there, and reused here under that
licence. See [NOTICE](../../../NOTICE).

**It is test data, not a fixture to edit.** Changing the file invalidates the
recorded measurement above, and a regression test whose expectation moves with
the code under test proves nothing.
