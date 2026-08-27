"""Shared YOLOv8 (5-class, P2 head) decode for both backends in this app.

Why one shared decoder for two very different backends (MLA/pyneat raw
tensors vs. CPU onnxruntime):

  We compiled yolov8_640_adamw_2.onnx to yolov8_640_adamw_2_mpk.tar.gz and
  inspected the compiled manifest (yolov8_640_adamw_2_mpk.json) directly.
  The final plugin in the pipeline (APU_22) has ONE output node,
  "APU_22/output0", sized 1,224,000 bytes = 9 * 34000 float32 -- i.e. the
  MLA pipeline's actual delivered output is already the fully-decoded
  (DFL softmax + multi-scale concat done on-device) [9, 34000] tensor,
  identical in shape and semantics to the raw ONNX graph's own "output0".
  That is exactly what onnxruntime hands back too. Same tensor, same decode.

  This was NOT obvious going in -- an earlier prototype in
  custom_yolov8_stream/main.py assumed the board would hand back 8 raw
  per-scale DFL tensors requiring manual softmax decode. That assumption
  was never confirmed against real hardware output (SSH access to the
  board was not available to verify it), and the compiled manifest
  contradicts it. decode_multiscale_dfl() below is kept ONLY as a
  defensive fallback in case a future SDK version changes what
  detess_dequant() returns -- normalize_to_flat_output() picks whichever
  path actually matches what came back, and prints which one it used.

9-channel layout per anchor (Ultralytics standard, confirmed by directly
inspecting yolov8_640_adamw_2.onnx's node graph): [cx, cy, w, h,
class_0..class_4] in the 640x640 model-input coordinate space.
"""

from __future__ import annotations

import numpy as np

NUM_CLASSES = 5
TOTAL_ANCHORS = 34000  # 160^2 + 80^2 + 40^2 + 20^2, strides [4, 8, 16, 32]
STRIDES = (4, 8, 16, 32)
GRID_SIDES = (160, 80, 40, 20)  # matches STRIDES order, largest grid first


def dfl_distance(logits: np.ndarray, reg_max: int) -> float:
    """Decode one DFL-encoded side: softmax over reg_max bins, weighted sum."""
    maxv = logits.max()
    e = np.exp(logits - maxv)
    denom = e.sum()
    if denom <= 0:
        return 0.0
    numer = np.dot(np.arange(reg_max, dtype=np.float32), e)
    return float(numer / denom)


def _channels_axis(arr: np.ndarray, expected: int) -> int:
    """Return which axis of a 2D array has length `expected` (channels)."""
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {arr.shape}")
    if arr.shape[0] == expected:
        return 0
    if arr.shape[1] == expected:
        return 1
    raise ValueError(f"no axis of shape {arr.shape} matches expected channel count {expected}")


def _to_channels_by_length(arr: np.ndarray) -> np.ndarray:
    """Normalize one raw per-scale tensor to (channels, H*W).

    Real hardware output (confirmed via --diagnose on the board) is NHWC:
    (1, 160, 160, 64), (1, 160, 160, 5), etc. -- channels LAST, matching
    the compiled manifest's own declared tensor_shapes/output_shapes
    (checked directly in yolov8_640_adamw_2_raw_mpk.json, e.g.
    MLA_0_ofm_unpack_transform's output_shapes are exactly this NHWC
    layout). An earlier version of this function assumed NCHW (channels
    first, matching raw ONNX conv output convention) -- that reshaped H
    into the channel slot, so cls tensors (5 channels) never matched
    num_classes at the shape[0] position pyneat actually reports it, every
    tensor got misclassified as a reg tensor, cls_tensors ended up empty,
    and decode crashed on cls_tensors[0] with "list index out of range"
    (exactly the crash --diagnose hit on real hardware). NCHW is kept as a
    fallback in case a different SDK build ever changes this.
    """
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"only batch=1 supported, got shape {arr.shape}")
        arr = arr[0]  # (H, W, C) on real hardware
    if arr.ndim == 3:
        # Channels-last (NHWC), unconditionally -- this is what real
        # hardware returns, confirmed via --diagnose. A prior version of
        # this tried to disambiguate NHWC vs NCHW per-tensor with a
        # "channel-like" heuristic (4, num_classes, or a multiple of 4 that
        # is >= 16); that broke because every one of our actual grid sizes
        # (160, 80, 40, 20) also satisfies "multiple of 4 and >= 16", so
        # the heuristic couldn't tell a 20x20 spatial dim from a real
        # channel count and silently misread the P5 scale. No heuristic
        # needed: trust the proven real layout instead of guessing.
        h, w, c = arr.shape
        arr = arr.transpose(2, 0, 1).reshape(c, h * w)  # NHWC -> (C, H*W)
    if arr.ndim != 2:
        raise ValueError(f"unexpected raw tensor rank after normalization: {arr.shape}")
    return arr


def decode_multiscale_dfl(tensors: list[np.ndarray], num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Fallback: reassemble output0-equivalent (4+num_classes, TOTAL_ANCHORS)
    from raw per-scale tensors, in case detess_dequant() ever returns those
    instead of the single fused tensor. Handles three raw layouts seen while
    building/debugging this app:
      (a) 4 reg tensors, each with reg_max*4 channels (reg_max >= 16, e.g.
          64) -- genuine un-decoded DFL logits, one per scale -- needs a
          real DFL softmax decode per scale, then grid-cell+stride offset
          to turn l,t,r,b distances into a model-space cx,cy,w,h box.
      (b) 1 reg tensor already merged across all scales (4, TOTAL_ANCHORS)
          + N per-scale class tensors needing only concatenation -- this is
          what the compiled manifest's intermediate stages suggested before
          the final on-device concat (kept here for robustness).
      (c) 4 reg tensors, each with exactly 4 channels, one per scale --
          this is what this app's tool-model-to-pipeline-compiled model
          (surgeon_yolov8_p2.py) actually returns on real hardware: the
          surgeon already collapses DFL to cx,cy,w,h INSIDE the ONNX graph
          (see its "bbox_version=2" Conv+Add reassembly), matching
          MODELS.md's documented `cxcywh_pixel` surgeoned bbox_format --
          "already-decoded [cx,cy,w,h] in model-space pixels, anchor+stride
          baked in". Treating this case like (a) is a real bug that was
          hit on hardware: reg_c=4 gives reg_max=4//4=1, and DFL-softmax
          over a single bin is mathematically always 0, silently
          collapsing every box to a zero-size point at its anchor
          center (confirmed: --diagnose showed box=(294,210,294,210),
          x1==x2 and y1==y2, on every detection). Distinguish (c) from
          (a) by reg_max itself: DFL always uses >=16 bins in practice;
          reg_max==1 is not a valid DFL bin count, so it's the signal
          that these are already-decoded boxes needing no further math.
    """
    # _to_channels_by_length already guarantees (channels, H*W) with
    # channels on axis 0 -- no further axis detection needed here.
    reg_tensors, cls_tensors = [], []
    for t in tensors:
        arr = _to_channels_by_length(np.asarray(t))
        if arr.shape[0] == num_classes:
            cls_tensors.append(arr)
        elif arr.shape[0] == 4 or (arr.shape[0] % 4 == 0 and arr.shape[0] >= 16):
            reg_tensors.append(arr)
        else:
            raise ValueError(f"tensor with unrecognized channel count {arr.shape[0]}")

    cls_tensors.sort(key=lambda a: -a.shape[1])  # finest grid (largest length) first
    cls_row = np.concatenate(cls_tensors, axis=1) if len(cls_tensors) > 1 else cls_tensors[0]
    if cls_row.shape[1] != TOTAL_ANCHORS:
        raise ValueError(f"class tensor total length {cls_row.shape[1]} != expected {TOTAL_ANCHORS}")

    if len(reg_tensors) == 1 and reg_tensors[0].shape[0] == 4 and reg_tensors[0].shape[1] == TOTAL_ANCHORS:
        reg_row = reg_tensors[0]  # (b) already merged + DFL-decoded
    elif all(reg.shape[0] == 4 for reg in reg_tensors) and len(reg_tensors) > 1:
        # (c) already-decoded per-scale cx,cy,w,h -- concatenate only, no
        # DFL, no grid/stride math (that's already baked in by the
        # surgeon). Order must match cls_tensors' finest-grid-first sort.
        reg_tensors.sort(key=lambda a: -a.shape[1])
        reg_row = np.concatenate(reg_tensors, axis=1)
        if reg_row.shape[1] != TOTAL_ANCHORS:
            raise ValueError(f"reg tensor total length {reg_row.shape[1]} != expected {TOTAL_ANCHORS}")
    else:
        # (a) genuine raw DFL logits -- decode + grid/stride offset.
        reg_tensors.sort(key=lambda a: -a.shape[1])
        decoded_parts = []
        for reg in reg_tensors:
            reg_c, n = reg.shape
            reg_max = reg_c // 4
            side = np.empty((4, n), dtype=np.float32)
            for i in range(4):
                block = reg[i * reg_max:(i + 1) * reg_max, :]
                for j in range(n):
                    side[i, j] = dfl_distance(block[:, j], reg_max)
            decoded_parts.append(side)
        reg_row = np.concatenate(decoded_parts, axis=1)
        if reg_row.shape[1] != TOTAL_ANCHORS:
            raise ValueError(f"reg tensor total length {reg_row.shape[1]} != expected {TOTAL_ANCHORS}")
        # Per-anchor grid-cell center + stride to turn ltrb distances into
        # a cx,cy,w,h box in 640x640 model space -- only valid for genuine
        # raw DFL logits (case (a)); case (c) is already in this form.
        offset = 0
        for side, stride in zip(GRID_SIDES, STRIDES):
            n = side * side
            ys, xs = np.divmod(np.arange(n), side)
            l, t, r, b = reg_row[0, offset:offset + n], reg_row[1, offset:offset + n], \
                reg_row[2, offset:offset + n], reg_row[3, offset:offset + n]
            cx = (xs + 0.5) * stride
            cy = (ys + 0.5) * stride
            x1, y1 = cx - l * stride, cy - t * stride
            x2, y2 = cx + r * stride, cy + b * stride
            reg_row[0, offset:offset + n] = (x1 + x2) / 2.0
            reg_row[1, offset:offset + n] = (y1 + y2) / 2.0
            reg_row[2, offset:offset + n] = x2 - x1
            reg_row[3, offset:offset + n] = y2 - y1
            offset += n

    return np.concatenate([reg_row, cls_row], axis=0)  # (4+num_classes, TOTAL_ANCHORS)


def normalize_to_flat_output(tensors: list[np.ndarray], num_classes: int = NUM_CLASSES) -> tuple[np.ndarray, str]:
    """Accepts whatever raw_tensors the board's detess_dequant() produced and
    returns (output, path) where output is (4+num_classes, TOTAL_ANCHORS)
    and path says which branch was taken -- log this on first run."""
    expected_channels = 4 + num_classes
    if len(tensors) == 1:
        arr = np.asarray(tensors[0])
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2:
            axis = _channels_axis(arr, expected_channels)
            out = arr if axis == 0 else arr.T
            if out.shape[1] == TOTAL_ANCHORS:
                return out.astype(np.float32), "fused-single-tensor"
    return decode_multiscale_dfl(tensors, num_classes).astype(np.float32), "multiscale-dfl-fallback"


def decode_flat_output(
    output: np.ndarray,
    num_classes: int,
    conf_thr: float,
    nms_iou: float,
    max_det: int,
    scale: float = 1.0,
    pad_left: float = 0.0,
    pad_top: float = 0.0,
) -> list[dict]:
    """output: (4+num_classes, N) -- per-anchor [cx, cy, w, h, class_scores].
    Returns boxes in ORIGINAL-frame pixel coordinates once scale/pad (from
    letterbox preprocessing) are supplied; pass scale=1, pad=0 to get boxes
    back in model-input (e.g. 640x640) space instead, for callers that
    apply their own resize-based rescaling afterward."""
    arr = output.T  # (N, 4+num_classes)
    boxes_raw = arr[:, :4]
    class_scores = arr[:, 4:4 + num_classes]
    if class_scores.size and (class_scores.max() > 1.0 or class_scores.min() < 0.0):
        class_scores = 1.0 / (1.0 + np.exp(-class_scores))  # guard: raw logits, not sigmoid yet

    best_cls = np.argmax(class_scores, axis=1)
    best_score = class_scores[np.arange(len(class_scores)), best_cls]
    keep_mask = best_score >= conf_thr
    if not np.any(keep_mask):
        return []

    boxes_raw = boxes_raw[keep_mask]
    best_cls = best_cls[keep_mask]
    best_score = best_score[keep_mask]

    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = (cx - w / 2 - pad_left) / scale
    y1 = (cy - h / 2 - pad_top) / scale
    x2 = (cx + w / 2 - pad_left) / scale
    y2 = (cy + h / 2 - pad_top) / scale

    candidates = [
        {"x1": float(x1[i]), "y1": float(y1[i]), "x2": float(x2[i]), "y2": float(y2[i]),
         "score": float(best_score[i]), "class_id": int(best_cls[i])}
        for i in range(len(best_score))
    ]
    candidates.sort(key=lambda d: d["score"], reverse=True)

    keep: list[dict] = []
    for cand in candidates:
        if len(keep) >= max_det:
            break
        suppressed = False
        for k in keep:
            if k["class_id"] != cand["class_id"]:
                continue
            xx1 = max(k["x1"], cand["x1"]); yy1 = max(k["y1"], cand["y1"])
            xx2 = min(k["x2"], cand["x2"]); yy2 = min(k["y2"], cand["y2"])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            area_k = max(0.0, k["x2"] - k["x1"]) * max(0.0, k["y2"] - k["y1"])
            area_c = max(0.0, cand["x2"] - cand["x1"]) * max(0.0, cand["y2"] - cand["y1"])
            union = area_k + area_c - inter
            if union > 0 and inter / union > nms_iou:
                suppressed = True
                break
        if not suppressed:
            keep.append(cand)
    return keep
