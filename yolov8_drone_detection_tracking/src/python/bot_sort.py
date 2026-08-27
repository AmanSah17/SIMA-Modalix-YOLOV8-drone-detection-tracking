"""Minimal BoT-SORT tracker (motion-only mode: Kalman filter + two-stage IoU
association, no ReID/appearance embedding branch). Pure numpy + scipy — no
torch, matching this board's lightweight Python environment.

Algorithm matches the public BoT-SORT / ByteTrack design:
  - Kalman filter per track, constant-velocity model on (cx, cy, aspect, h).
  - Two-stage association per frame: high-score detections matched first
    against all tracks (IoU cost), then low-score detections matched against
    remaining unmatched tracks (recovers occluded/blurred objects).
  - Unmatched tracks are kept "lost" for `max_missing` frames before removal.
  - A track is only reported once it has `min_hits` confirmed associations.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over state [cx, cy, aspect, h, vcx, vcy, vaspect, vh]."""

    _next_id = 1

    def __init__(self, xyxy: np.ndarray, score: float, class_id: int) -> None:
        cx, cy, a, h = self._xyxy_to_cah(xyxy)
        self.mean = np.array([cx, cy, a, h, 0, 0, 0, 0], dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32) * 10.0
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        self.track_id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.hits = 1
        self.time_since_update = 0
        self.score = score
        self.class_id = class_id
        self.confirmed = False

    @staticmethod
    def _xyxy_to_cah(xyxy: np.ndarray):
        x1, y1, x2, y2 = xyxy
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return cx, cy, w / h, h

    @staticmethod
    def _cah_to_xyxy(cx, cy, a, h):
        w = a * h
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

    def predict(self) -> None:
        std_pos = [
            self._std_weight_position * self.mean[3],
            self._std_weight_position * self.mean[3],
            1e-2,
            self._std_weight_position * self.mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * self.mean[3],
            self._std_weight_velocity * self.mean[3],
            1e-5,
            self._std_weight_velocity * self.mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = 1.0

        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + motion_cov
        self.time_since_update += 1

    def update(self, xyxy: np.ndarray, score: float, class_id: int) -> None:
        cx, cy, a, h = self._xyxy_to_cah(xyxy)
        measurement = np.array([cx, cy, a, h], dtype=np.float32)

        std_pos = [
            self._std_weight_position * self.mean[3],
            self._std_weight_position * self.mean[3],
            1e-1,
            self._std_weight_position * self.mean[3],
        ]
        innovation_cov = np.diag(np.square(std_pos)).astype(np.float32)

        H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            H[i, i] = 1.0

        projected_cov = H @ self.covariance @ H.T + innovation_cov
        kalman_gain = self.covariance @ H.T @ np.linalg.inv(projected_cov)
        innovation = measurement - H @ self.mean

        self.mean = self.mean + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ H @ self.covariance

        self.hits += 1
        self.time_since_update = 0
        self.score = score
        self.class_id = class_id

    def to_xyxy(self) -> np.ndarray:
        return self._cah_to_xyxy(*self.mean[:4])


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    w = np.clip(xx2 - xx1, 0, None)
    h = np.clip(yy2 - yy1, 0, None)
    inter = w * h
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _greedy_match(cost: np.ndarray, thresh: float):
    """Hungarian assignment, then drop pairs above the cost threshold."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    row_idx, col_idx = linear_sum_assignment(cost)
    matches, unmatched_rows, unmatched_cols = [], [], []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= thresh:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)
    unmatched_rows = [r for r in range(cost.shape[0]) if r not in matched_rows]
    unmatched_cols = [c for c in range(cost.shape[1]) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


class BotSortTracker:
    """Frame-by-frame BoT-SORT tracker (motion-only). Call update() once per frame."""

    def __init__(
        self,
        high_score_thresh: float = 0.5,
        low_score_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        iou_match_thresh: float = 0.3,
        max_missing: int = 30,
        min_hits: int = 2,
    ) -> None:
        self.high_score_thresh = high_score_thresh
        self.low_score_thresh = low_score_thresh
        self.new_track_thresh = new_track_thresh
        self.iou_match_thresh = iou_match_thresh
        self.max_missing = max_missing
        self.min_hits = min_hits
        self.tracks: list[KalmanBoxTracker] = []

    def update(self, detections: list[dict]) -> list[dict]:
        """detections: list of {x1,y1,x2,y2,score,class_id}. Returns the same
        dicts for currently-confirmed tracks, each with an added 'track_id'."""
        for t in self.tracks:
            t.predict()

        boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=np.float32) \
            if detections else np.zeros((0, 4), dtype=np.float32)
        scores = np.array([d["score"] for d in detections], dtype=np.float32) \
            if detections else np.zeros((0,), dtype=np.float32)

        high_idx = np.where(scores >= self.high_score_thresh)[0]
        low_idx = np.where((scores >= self.low_score_thresh) & (scores < self.high_score_thresh))[0]

        track_boxes = np.array([t.to_xyxy() for t in self.tracks], dtype=np.float32) \
            if self.tracks else np.zeros((0, 4), dtype=np.float32)

        unmatched_tracks = list(range(len(self.tracks)))

        # Stage 1: high-score detections vs all tracks.
        if len(high_idx) and len(self.tracks):
            iou = iou_batch(boxes[high_idx], track_boxes)
            cost = 1.0 - iou
            matches, um_det_local, um_trk = _greedy_match(cost, 1.0 - self.iou_match_thresh)
            matched_det = set()
            for det_local, trk_i in matches:
                det_i = high_idx[det_local]
                d = detections[det_i]
                self.tracks[trk_i].update(boxes[det_i], scores[det_i], d["class_id"])
                matched_det.add(det_i)
            unmatched_tracks = um_trk
            unmatched_high = [high_idx[i] for i in um_det_local]
        else:
            matched_det = set()
            unmatched_high = list(high_idx)

        # Stage 2: low-score detections vs tracks still unmatched (recovers occlusions).
        if len(low_idx) and unmatched_tracks:
            remaining_boxes = track_boxes[unmatched_tracks]
            iou = iou_batch(boxes[low_idx], remaining_boxes)
            cost = 1.0 - iou
            matches, _um_det_local, um_trk_local = _greedy_match(cost, 1.0 - self.iou_match_thresh)
            still_unmatched = []
            matched_local_trk = set()
            for det_local, trk_local in matches:
                det_i = low_idx[det_local]
                trk_i = unmatched_tracks[trk_local]
                d = detections[det_i]
                self.tracks[trk_i].update(boxes[det_i], scores[det_i], d["class_id"])
                matched_det.add(det_i)
                matched_local_trk.add(trk_local)
            unmatched_tracks = [unmatched_tracks[i] for i in range(len(unmatched_tracks)) if i not in matched_local_trk]

        # Spawn new tracks from confident unmatched high-score detections.
        for det_i in unmatched_high:
            if det_i in matched_det:
                continue
            if scores[det_i] >= self.new_track_thresh:
                self.tracks.append(KalmanBoxTracker(boxes[det_i], scores[det_i], detections[det_i]["class_id"]))

        # Age out tracks that have been missing too long.
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_missing]

        results = []
        for t in self.tracks:
            if t.time_since_update > 0:
                continue
            if t.hits < self.min_hits:
                continue
            x1, y1, x2, y2 = t.to_xyxy()
            results.append({
                "track_id": t.track_id,
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "score": float(t.score), "class_id": int(t.class_id),
            })
        return results
