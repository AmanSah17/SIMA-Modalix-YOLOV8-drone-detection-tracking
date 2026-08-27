"""RTSP -> YOLOv8 (5-class, P2 head, MLA-accelerated) -> BoT-SORT -> Insight.

Backend: yolov8_drone_p2_mpk.tar.gz -- compiled via SiMa's official
tool-model-to-pipeline (custom surgeon_yolov8_p2.py, since the stock
surgeon assumes 3 scales at model.22; this model has 4 at model.28) --
running on the MLA accelerator via pyneat. Manual decode (this file, via
yolo_decode.py), NOT on-device decode: confirmed on real hardware that
pyneat.BoxDecodeType.YoloV8 exists, but it's a fixed 3-scale kernel
(reg_p3/p4/p5 only) that explicitly rejects this model's 4-scale P2 head
-- "YOLO BoxDecode fallback could not validate grouped-by-role raw DFL
output order. Expected [reg_p3, reg_p4, reg_p5, ...]" was the exact
on-device error. Manual decode is the only working path for this
architecture. model.path/model.labels in config.yaml must point at our
5-class artifacts (see src/common/config.yaml).

Decode: see yolo_decode.py's module docstring for why this reads the raw
per-frame tensor as a single already-decoded (4+num_classes, 34000) array
rather than 8 raw per-scale DFL tensors -- that was confirmed by tracing
the compiled MPK's own manifest (yolov8_640_adamw_2_mpk.json) end to end,
not assumed. normalize_to_flat_output() still carries a defensive fallback
in case a different SDK build ever hands back the raw per-scale tensors
instead; --diagnose prints which path actually gets used on real hardware
the first time this runs (needed: nothing here could be verified against
the live board -- SSH access to it was not available while writing this).

preprocessing note: on-device preprocess below is left as plain resize (no
letterbox pad), matching compile_yolov8_custom.py's PTQ calibration
preprocessing exactly (cv2.resize + /255, no aspect-ratio padding). Boxes
decoded here come back in 640x640 model space and are rescaled to the
source frame with separate x/y factors (non-uniform, matching that plain
resize) rather than a letterbox unscale -- see build_metadata_boxes().

Usage:
  python3 main.py --config ../common/config.yaml --diagnose   # inspect raw tensor(s) once, exit
  python3 main.py --config ../common/config.yaml               # run detection + tracking
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from bot_sort import BotSortTracker
from yolo_decode import NUM_CLASSES, TOTAL_ANCHORS, decode_flat_output, normalize_to_flat_output

cv2 = None
pyneat = None


def load_runtime_dependencies() -> None:
    global cv2, pyneat
    if pyneat is not None:
        return
    import cv2 as cv2_module
    import pyneat as pyneat_module
    cv2 = cv2_module
    pyneat = pyneat_module


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels: list[str]
    source_url: str
    source_codec: str = "h264"
    latency_ms: int = 200
    tcp: bool = True
    infer_size: int = 640
    num_classes: int = NUM_CLASSES
    score_threshold: float = 0.30
    nms_iou: float = 0.50
    max_detections: int = 100
    insight_host: str = "127.0.0.1"
    video_port: int = 9000
    metadata_port: int = 9100
    save_video_path: str = ""
    tracker_high_score: float = 0.5
    tracker_low_score: float = 0.1
    tracker_new_track: float = 0.6
    tracker_iou_match: float = 0.3
    tracker_max_missing: int = 30
    tracker_min_hits: int = 2


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = raw.get("model", {}) or {}
    source = raw.get("source", {}) or {}
    runtime = raw.get("runtime", {}) or {}
    decode = raw.get("decode", {}) or {}
    output = raw.get("output", {}) or {}
    insight = output.get("insight", {}) or {}
    tracker = raw.get("tracker", {}) or {}

    labels_path = Path(model.get("labels", ""))
    labels = [l.strip() for l in labels_path.read_text().splitlines() if l.strip()] if labels_path.is_file() else []

    cfg = AppConfig(
        model_path=model.get("path", ""),
        labels=labels,
        source_url=source.get("url", ""),
        source_codec=source.get("codec", "h264"),
        latency_ms=int(source.get("latency_ms", 200)),
        tcp=bool(source.get("tcp", True)),
        infer_size=int(runtime.get("infer_size", 640)),
        num_classes=int(runtime.get("num_classes", len(labels) or NUM_CLASSES)),
        score_threshold=float(decode.get("score_threshold", 0.30)),
        nms_iou=float(decode.get("nms_iou", 0.50)),
        max_detections=int(decode.get("max_detections", 100)),
        insight_host=insight.get("host", "127.0.0.1"),
        video_port=int(insight.get("video_port", 9000)),
        metadata_port=int(insight.get("metadata_port", 9100)),
        save_video_path=output.get("save_video_path", ""),
        tracker_high_score=float(tracker.get("high_score_thresh", 0.5)),
        tracker_low_score=float(tracker.get("low_score_thresh", 0.1)),
        tracker_new_track=float(tracker.get("new_track_thresh", 0.6)),
        tracker_iou_match=float(tracker.get("iou_match_thresh", 0.3)),
        tracker_max_missing=int(tracker.get("max_missing", 30)),
        tracker_min_hits=int(tracker.get("min_hits", 2)),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    if not cfg.model_path:
        raise ValueError("model.path must be set (point it at yolov8_640_adamw_2_mpk.tar.gz)")
    if not cfg.labels:
        raise ValueError("model.labels must point at a non-empty labels file")
    if len(cfg.labels) != cfg.num_classes:
        raise ValueError(
            f"labels file has {len(cfg.labels)} entries but runtime.num_classes={cfg.num_classes} "
            "-- these must match exactly or class IDs will be mislabeled"
        )
    if not cfg.source_url:
        raise ValueError("source.url must be set")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if not 0.0 <= cfg.score_threshold <= 1.0:
        raise ValueError("decode.score_threshold must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("decode.nms_iou must be between 0 and 1")


# ---------------------------------------------------------------------------
# Raw tensor extraction from a pyneat sample.
# ---------------------------------------------------------------------------

def tensor_to_numpy(tensor) -> np.ndarray:
    dtype_map = {
        pyneat.TensorDType.UInt8: np.uint8,
        pyneat.TensorDType.Int8: np.int8,
        pyneat.TensorDType.UInt16: np.uint16,
        pyneat.TensorDType.Int16: np.int16,
        pyneat.TensorDType.Int32: np.int32,
        pyneat.TensorDType.Float32: np.float32,
        pyneat.TensorDType.Float64: np.float64,
    }
    np_dtype = dtype_map[tensor.dtype]
    shape = tuple(int(x) for x in tensor.shape)
    arr = np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=np_dtype)
    return arr.reshape(shape) if shape else arr


def iter_tensors(sample):
    if sample.kind == pyneat.SampleKind.Tensor:
        if sample.tensor is None:
            raise RuntimeError("tensor sample missing payload")
        yield sample.tensor
    elif sample.kind == pyneat.SampleKind.TensorSet:
        yield from sample.tensors
    elif sample.kind == pyneat.SampleKind.Bundle:
        for f in sample.fields:
            yield from iter_tensors(f)
    else:
        raise RuntimeError(f"unexpected sample kind: {sample.kind}")


def find_field(sample, label: str):
    if getattr(sample, "stream_label", "") == label:
        return sample
    for f in getattr(sample, "fields", []):
        found = find_field(f, label)
        if found is not None:
            return found
    return None


def first_tensor_from_sample(sample):
    if sample is None:
        return None
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return sample.tensors[0]
    for f in getattr(sample, "fields", []):
        t = first_tensor_from_sample(f)
        if t is not None:
            return t
    return None


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def frame_bgr_from_sample(sample) -> np.ndarray:
    field = find_field(sample, "frame") or sample
    tensor = first_tensor_from_sample(field)
    if tensor is None:
        raise RuntimeError("joined sample has no frame tensor")
    if not tensor.is_nv12():
        raise RuntimeError("expected NV12 frame tensor")
    width, height = tensor_dim(tensor, "width"), tensor_dim(tensor, "height")
    payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
    expected = width * height * 3 // 2
    nv12 = payload[:expected].reshape((height * 3 // 2, width))
    return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))


# ---------------------------------------------------------------------------
# Pipeline graph.
# ---------------------------------------------------------------------------

def probe_rtsp(url: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP frame size")
    return width, height, fps or 30


def make_rtsp_source(cfg: AppConfig, width: int, height: int, fps: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.source_url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.source_fps = fps
    opt.codec = pyneat.RtspCodec.H264 if cfg.source_codec == "h264" else pyneat.RtspCodec.H265
    opt.payload_type = 96
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = width
    opt.fallback_h264_height = height
    caps = opt.output_caps
    caps.enable = True
    caps.format = pyneat.Format.NV12
    caps.width = width
    caps.height = height
    caps.fps = fps
    caps.memory = pyneat.CapsMemory.Any
    return pyneat.groups.rtsp_decoded_input(opt)


def make_still_image_source(image_path: str):
    """Feed one static image through the exact same graph a live RTSP frame
    would go through (StillImageInput carries InputRole::Source, same as
    the RTSP source -- see core-src's include/nodes/io/StillImageInput.h),
    for deterministic single-image testing instead of depending on the live
    stream. Not verified on real hardware -- if the plain-int args below
    don't match what pyneat.nodes.still_image_input actually expects
    (the C++ signature takes strong-typed wrapper structs: ContentWidth,
    ContentHeight, EncodeWidth, EncodeHeight, FramesPerSecond -- pybind11
    may or may not accept plain ints for these), the resulting TypeError
    will show the real accepted signature, same as the q_scale fix did.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    height, width = img.shape[:2]
    source = pyneat.nodes.still_image_input(image_path, width, height, width, height, 1)
    return source, width, height, 1


def make_model(cfg: AppConfig, source_width: int, source_height: int):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    # input_max_width/height is a capacity bound on the incoming SOURCE
    # frame, not the model's target inference size -- the preprocessor
    # resizes down to the model's declared input (640x640) internally.
    # Setting this to cfg.infer_size (640) instead of the actual source
    # resolution rejected every real frame from the 1920x1080 stream with
    # "input width 1920 exceeds max_input_width 640". Matches the pattern
    # in SiMa's single-stream-object-detector reference example, which
    # sets these from the probed stream/image dimensions.
    if source_width > 0 and source_height > 0:
        opt.preprocess.input_max_width = source_width
        opt.preprocess.input_max_height = source_height
    # No opt.decode_type set on purpose: our model isn't YOLO26, so the
    # on-device BoxDecodeType presets don't apply. We take the model's raw
    # output tensor via a post-inference dequant stage (see make_post_stage)
    # and decode it ourselves below.
    return pyneat.Model(cfg.model_path, opt)


#: Per-output-tensor (scale, zero_point) for the OLDER, MLA-tessellated
#: compile (yolov8_640_adamw_2_raw_mpk.tar.gz built WITHOUT
#: --no-mla-tessellation), read out of that manifest's dequantize_2..
#: dequantize_9 plugins (channel_params), in declared-output order: P2
#: box, P2 cls, P3 box, P3 cls, P4 box, P4 cls, P5 box, P5 cls. Not used
#: by the current default config (see config.yaml) -- kept only as
#: make_post_stage()'s fallback in case a future compile lands back in
#: that no-detessellate-stage situation.
RAW_OUTPUT_DEQUANT_PARAMS = [
    (10.692065066085283, -56),   # cv2.0 (P2 box)
    (7.6379224317207575, 118),   # cv3.0 (P2 cls)
    (12.101162090065294, -64),   # cv2.1 (P3 box)
    (7.008674465006425, 127),    # cv3.1 (P3 cls)
    (11.4979949699918, -32),     # cv2.2 (P4 box)
    (6.292981458036563, 127),    # cv3.2 (P4 cls)
    (22.732827507220783, 18),    # cv2.3 (P5 box)
    (6.796350507633491, 127),    # cv3.3 (P5 cls)
]


def make_post_stage(model, verbose: bool = False):
    """Get the model's raw per-scale tensors off the accelerator.

    The default compile (compiled with --no-mla-tessellation, see
    config.yaml's comment on model.path) produces a standard route with an
    explicit detessellate stage ahead of each dequantize stage -- verified
    directly in that build's manifest (8 "detessellation_transform" plugins,
    one per output, each immediately before its "dequantization_transform"
    plugin). That's exactly the route every official reference example
    (yolov8-instance-segmenter, etc.) assumes when it calls
    detess_dequant(DetessDequantOptions(model)) -- so that's the primary
    path here too.

    Fallback: an EARLIER compile of this same graph-surgeried model, done
    WITH the default MLA output tessellation
    (compiled_yolov8_640_adamw_2_raw/, not the _notess/ one this config
    points at), produced a route with NO separate detessellate stage
    (MLA_0_ofm_unpack_transform already unpacked straight to per-scale
    tensors), which made detess_dequant() raise "resolved model route does
    not contain DetessDequant" -- kept here in case that variant is ever
    used again. That path needed per-tensor DequantOptions
    (stage_id/element_name-selected, q_scale/q_zp as single floats each,
    confirmed via the actual pybind signature error on real hardware) that
    was never fully wired up, since the primary path below removed the
    need for it. If you hit that error again, this will still fail --
    paste the error and it can be finished properly instead of guessed at.
    """
    try:
        return pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model))
    except Exception as exc:  # exact pyneat exception type for this error is unconfirmed
        if "does not contain DetessDequant" not in str(exc):
            raise
        print(f"[dequant] detess_dequant() route mismatch ({exc}); "
              "falling back to per-tensor Dequant (unverified, see make_post_stage docstring)", file=sys.stderr)

    opts = pyneat.DequantOptions(model)
    if verbose:
        print(f"[dequant] DequantOptions attributes: {[a for a in dir(opts) if not a.startswith('_')]}",
              file=sys.stderr)
    scales = [s for s, _ in RAW_OUTPUT_DEQUANT_PARAMS]
    zero_points = [zp for _, zp in RAW_OUTPUT_DEQUANT_PARAMS]
    try:
        opts.q_scale = scales
        opts.q_zp = zero_points
    except Exception as exc:
        raise RuntimeError(
            f"fallback per-tensor dequant also failed to set q_scale/q_zp as lists ({exc}). "
            f"DequantOptions attributes: {[a for a in dir(opts) if not a.startswith('_')]}. "
            "This model needs one Dequant node per output tensor (stage_id-selected), not "
            "one node for the whole bundle -- paste this error back to finish it properly."
        ) from exc
    return pyneat.nodes.dequant(opts)


def build_video_graph(cfg: AppConfig, width: int, height: int, fps: int):
    sender_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, fps)
    sender_options.host = cfg.insight_host
    sender_options.channel = 0
    sender_options.video_port_base = cfg.video_port
    sender_options.encoder.bitrate_kbps = 1000
    graph = pyneat.Graph("video")
    graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(sender_options))
    return graph, sender_options.video_port


def build_pipeline(cfg: AppConfig, image_path: str | None = None):
    if image_path:
        source, width, height, fps = make_still_image_source(image_path)
    else:
        width, height, fps = probe_rtsp(cfg.source_url)
        source = make_rtsp_source(cfg, width, height, fps)
    model = make_model(cfg, width, height)
    save_video = bool(cfg.save_video_path)

    branches = ["video", "model", "frame"] if save_video else ["video", "model"]
    branch = pyneat.graphs.branch("source", branches)
    video_graph, video_port = build_video_graph(cfg, width, height, fps)

    # NOTE: tried throttling this to every_frame(4) to cut Python-side load
    # (this model's route lands on the multiscale-dfl-fallback decode path --
    # 8 tensors/frame, ~9.3MB, DFL softmax decode over ~34000 anchors -- and
    # every_frame(1) alone backpressures: "[ERR] [runtime.element_failed]
    # GraphRun: pipeline input backpressure timeout (seg=2, edge_queue=3,
    # push_timeout_ms=5000)"). That made it WORSE -- immediate
    # output_pool_exhausted at detessdequant_9 instead -- so EveryFrame(N)
    # here evidently doesn't stop the MLA/detess stages producing (and
    # retaining) a buffer for every decoded frame regardless of N; it only
    # filters what reaches Python, so skipped frames' buffers just pile up
    # unreleased. Left at 1 (matches the working --diagnose run); the actual
    # fix attempted below is RunOptions.queue_depth.
    raw_tensor_rate = 1
    model_graph = pyneat.Graph("model")
    model_graph.add(pyneat.nodes.input("model"))
    model_graph.add(model.preprocess())
    model_graph.add(model.inference())
    model_graph.add(make_post_stage(model, verbose=True))
    model_graph.add(pyneat.nodes.output("raw_tensors", pyneat.OutputOptions.every_frame(raw_tensor_rate)))

    # Per-edge backpressure control -- the piece our earlier attempts (only
    # OutputOptions.every_frame(N) on the terminal "raw_tensors"/"frame"
    # output nodes, only RunOptions.queue_depth globally) were missing.
    # SiMa's official multi-stream reference (yolo26-tiny-drone-tracker,
    # src/python/main.py's realtime_link()) caps in-flight buffers on the
    # specific edge feeding the model/detector graph via a
    # GraphLinkOptions passed as graph.connect()'s 3rd arg -- every one of
    # OUR graph.connect() calls used the bare 2-arg form, so nothing ever
    # bounded how many samples could queue up between the branch and
    # model_graph while Python's multiscale-dfl-fallback decode fell
    # behind real time, which is what actually produced both the
    # "output_pool_exhausted" and "pipeline input backpressure timeout"
    # errors below. Not yet re-verified on hardware after this edit --
    # SSH access to the board was withdrawn mid-session (see chat) --
    # if pyneat.GraphLinkOptions/GraphLinkPolicy don't match this shape,
    # this will raise AttributeError; paste that back to fix the field
    # names against the actual pybind signature.
    model_link = pyneat.GraphLinkOptions()
    model_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    model_link.queue_depth = 4
    model_link.max_inflight_per_stream = 4
    model_link.max_inflight_total = 4
    model_link.stream_id = "stream0"

    graph = pyneat.Graph()
    graph.connect(source, branch)
    graph.connect(branch, video_graph)
    graph.connect(branch, model_graph, model_link)

    output_name = "raw_tensors"
    if save_video:
        frame_graph = pyneat.Graph("frame")
        # every_frame(1) (emit every decoded frame as a zero-copy NV12
        # tensor) exhausts the pipeline's output buffer pool once cv2's
        # NV12->BGR convert + mp4v write can't keep up with realtime
        # 1920x1080@30 -- confirmed on hardware:
        # "[ERR] [resource.output_pool_exhausted] ... Stage: detessdequant_9".
        # SiMa's own single-stream-object-detector reference hits the same
        # combine(frame, detections, ByFrame) shape and uses every_frame(4)
        # on BOTH sides for exactly this reason -- matched here.
        frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))
        joined = pyneat.graphs.combine(
            ["frame", "raw_tensors"], "detector_output", pyneat.CombinePolicy.ByFrame
        )
        frame_link = pyneat.GraphLinkOptions()
        frame_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
        frame_link.queue_depth = 4
        frame_link.max_inflight_per_stream = 4
        frame_link.max_inflight_total = 4
        frame_link.stream_id = "stream0"
        graph.connect(branch, frame_graph, frame_link)
        graph.connect(frame_graph, joined)
        graph.connect(model_graph, joined)
        output_name = "detector_output"

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    # Matches yolo26-tiny-drone-tracker's build_run_options() (the official
    # multi-stream reference) exactly -- global queue_depth=4, with the
    # actual backpressure bound now coming from model_link's per-edge
    # max_inflight_* above, not this value.
    run_options.queue_depth = 4
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    run = graph.build(run_options)

    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.channel = 0
    metadata_options.metadata_port_base = cfg.metadata_port
    metadata_sender = pyneat.MetadataSender(metadata_options)

    print(
        f"source={image_path or cfg.source_url} stream={width}x{height}@{fps} model={cfg.model_path} "
        f"insight={cfg.insight_host} video={video_port} metadata={metadata_sender.metadata_port()} "
        f"save_video={cfg.save_video_path or '(disabled)'}",
        flush=True,
    )
    return run, metadata_sender, width, height, fps, output_name


def run_diagnose(cfg: AppConfig, image_path: str | None = None) -> int:
    """Pull one frame (from RTSP, or a static image if --image is given),
    show the raw tensor(s) and which decode path they resolve to, then
    exit -- run this FIRST against real hardware, since nothing here could
    be verified against the live board while writing it (see module
    docstring)."""
    run, _metadata_sender, _w, _h, _fps, output_name = build_pipeline(cfg, image_path=image_path)
    try:
        sample = run.pull(output_name, 20000)
        if sample is None:
            print("[diagnose] timed out waiting for the first frame", file=sys.stderr)
            return 1
        tensor_field = find_field(sample, "raw_tensors") or sample
        tensors = [tensor_to_numpy(t) for t in iter_tensors(tensor_field)]
        print(f"[diagnose] received {len(tensors)} raw tensor(s):")
        for i, arr in enumerate(tensors):
            print(f"  [{i}] shape={arr.shape} dtype={arr.dtype} min={arr.min():.3f} max={arr.max():.3f}")
        try:
            normalized, path = normalize_to_flat_output(tensors, cfg.num_classes)
            print(f"\n[diagnose] decode path selected: {path}")
            print(f"[diagnose] normalized output shape: {normalized.shape} (expected (9, {TOTAL_ANCHORS}))")
            dets = decode_flat_output(normalized, cfg.num_classes, cfg.score_threshold, cfg.nms_iou, cfg.max_detections)
            print(f"[diagnose] {len(dets)} detection(s) above score_threshold={cfg.score_threshold} on this frame")
            for d in dets[:10]:
                name = cfg.labels[d["class_id"]] if 0 <= d["class_id"] < len(cfg.labels) else f"class_{d['class_id']}"
                print(f"    {name} score={d['score']:.3f} box=({d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f})")
            if not dets:
                # threshold=0 bypasses the cutoff so we can see the actual
                # best candidates instead of just "0 detections" -- tells
                # us whether this is a genuinely empty frame or a
                # near-miss (candidate present, just under threshold).
                top = decode_flat_output(normalized, cfg.num_classes, 0.0, cfg.nms_iou, 5)
                print(f"[diagnose] no detections cleared the threshold -- top {len(top)} candidate(s) regardless of threshold:")
                for d in top:
                    name = cfg.labels[d["class_id"]] if 0 <= d["class_id"] < len(cfg.labels) else f"class_{d['class_id']}"
                    print(f"    {name} score={d['score']:.3f} box=({d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f})")
        except Exception as exc:
            print(f"\n[diagnose] decode FAILED: {exc}", file=sys.stderr)
            print("[diagnose] compare the raw shapes above against yolo_decode.py's assumptions.", file=sys.stderr)
            return 1
        return 0
    finally:
        run.close()


def build_metadata_boxes(tracks: list[dict], labels: list[str], scale_x: float, scale_y: float) -> list[dict]:
    out = []
    for t in tracks:
        class_id = t["class_id"]
        out.append({
            "id": f"track_{t['track_id']}",
            "label": labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}",
            "confidence": t["score"],
            "track_id": t["track_id"],
            "bbox": [
                t["x1"] * scale_x, t["y1"] * scale_y,
                (t["x2"] - t["x1"]) * scale_x, (t["y2"] - t["y1"]) * scale_y,
            ],
        })
    return out


def draw_tracks_and_fps(frame_bgr: np.ndarray, tracks: list[dict], labels: list[str], fps: float) -> None:
    for t in tracks:
        x1, y1, x2, y2 = int(t["x1"]), int(t["y1"]), int(t["x2"]), int(t["y2"])
        class_id = t["class_id"]
        name = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"#{t['track_id']} {name} {t['score']:.2f}"
        cv2.putText(frame_bgr, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


def run_pipeline(cfg: AppConfig) -> int:
    run, metadata_sender, frame_w, frame_h, source_fps, output_name = build_pipeline(cfg)
    tracker = BotSortTracker(
        high_score_thresh=cfg.tracker_high_score,
        low_score_thresh=cfg.tracker_low_score,
        new_track_thresh=cfg.tracker_new_track,
        iou_match_thresh=cfg.tracker_iou_match,
        max_missing=cfg.tracker_max_missing,
        min_hits=cfg.tracker_min_hits,
    )
    scale_x = frame_w / cfg.infer_size
    scale_y = frame_h / cfg.infer_size
    processed = 0
    decode_path_logged = False

    writer = None
    if cfg.save_video_path:
        out_path = Path(cfg.save_video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, source_fps, (frame_w, frame_h))
        if not writer.isOpened():
            print(f"[ERR] failed to open video writer at {out_path}", file=sys.stderr)
            writer = None
        else:
            print(f"[video] writing annotated output to {out_path}", flush=True)

    fps_window_start = time.perf_counter()
    fps_window_frames = 0
    measured_fps = 0.0

    try:
        while True:
            sample = run.pull(output_name, 20000)
            if sample is None:
                print("[warn] timed out waiting for a frame", file=sys.stderr)
                continue

            tensor_field = find_field(sample, "raw_tensors") if writer is not None else sample
            tensors = [tensor_to_numpy(t) for t in iter_tensors(tensor_field or sample)]
            try:
                normalized, path = normalize_to_flat_output(tensors, cfg.num_classes)
                if not decode_path_logged:
                    print(f"[decode] using path: {path}", flush=True)
                    decode_path_logged = True
                detections = decode_flat_output(
                    normalized, cfg.num_classes, cfg.score_threshold, cfg.nms_iou, cfg.max_detections,
                )
            except Exception as exc:
                print(f"[decode error] {exc}", file=sys.stderr)
                continue

            tracks = tracker.update(detections)
            metadata_boxes = build_metadata_boxes(tracks, cfg.labels, scale_x, scale_y)

            timestamp_ms = int(sample.pts_ns // 1_000_000) if getattr(sample, "pts_ns", -1) >= 0 else -1
            frame_id = str(sample.frame_id) if getattr(sample, "frame_id", -1) >= 0 else ""
            metadata_sender.send_metadata(
                # Insight's frontend (drawing.js: window.drawStrategies) only
                # recognizes "tracking" as the type, with tracks under a
                # "tracks" key -- not "object-tracking"/"objects", which it
                # silently ignores (no error, just nothing drawn).
                "tracking",
                json.dumps({"tracks": metadata_boxes}, separators=(",", ":")),
                timestamp_ms,
                frame_id,
            )

            processed += 1
            fps_window_frames += 1
            elapsed = time.perf_counter() - fps_window_start
            if elapsed >= 1.0:
                measured_fps = fps_window_frames / elapsed
                fps_window_frames = 0
                fps_window_start = time.perf_counter()

            if writer is not None:
                try:
                    frame_bgr = frame_bgr_from_sample(sample)
                    scaled_tracks = [
                        {**t, "x1": t["x1"] * scale_x, "y1": t["y1"] * scale_y,
                         "x2": t["x2"] * scale_x, "y2": t["y2"] * scale_y}
                        for t in tracks
                    ]
                    draw_tracks_and_fps(frame_bgr, scaled_tracks, cfg.labels, measured_fps)
                    writer.write(frame_bgr)
                except Exception as exc:
                    print(f"[video write error] {exc}", file=sys.stderr)

            # Per-frame status (was every 100 frames -- console print throttle
            # only, metadata was already being sent to Insight every frame
            # regardless; this doesn't change actual processing/tracking).
            track_summary = ", ".join(
                f"{cfg.labels[t['class_id']] if 0 <= t['class_id'] < len(cfg.labels) else t['class_id']}"
                f"#{t['track_id']}:{t['score']:.2f}"
                for t in tracks
            ) or "none"
            print(f"[frame {processed}] tracks={len(tracks)} fps={measured_fps:.1f} [{track_summary}]", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        run.close()
        if writer is not None:
            writer.release()
    print(f"processed={processed}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--diagnose", action="store_true", help="Print raw tensor shapes + decode path for one frame, then exit.")
    parser.add_argument("--image", type=str, default=None,
                        help="Run --diagnose against one static image file instead of the RTSP source "
                             "(e.g. one of the 166 calibration frames on this board). Implies --diagnose.")
    args = parser.parse_args(argv)

    load_runtime_dependencies()
    try:
        cfg = load_config(args.config)
    except (ValueError, OSError) as exc:
        print(f"[ERR] invalid config: {exc}", file=sys.stderr)
        return 1

    try:
        if args.diagnose or args.image:
            return run_diagnose(cfg, image_path=args.image)
        return run_pipeline(cfg)
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
