#!/usr/bin/env python3
"""
YOLO Detection Benchmark — N=10 trials
Run on Raspberry Pi 4 to collect publishable metrics.

Measures:
  1. Per-frame inference time (ms)  — mean ± std over 100 frames per trial
  2. End-to-end pipeline FPS        — including capture + preprocess + detect + draw
  3. Detection accuracy (confidence scores, hit/miss per frame)

Usage:
  python3 benchmark_yolo.py              # 10 trials, 100 frames each
  python3 benchmark_yolo.py --trials 5   # 5 trials
  python3 benchmark_yolo.py --headless   # no display window
"""

import os
import sys
import time
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime

# Display setup for Pi
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent / "src"))

from capture_threaded import ThreadedCamera as CameraCapture
from detector import TFLiteDetector

PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "model" / "yolov8n-fp16.tflite"
RESULTS_DIR = PROJECT_ROOT / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Pipeline config — matches main.py exactly
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45
CAMERA_RESOLUTION = (320, 320)
CAMERA_FRAMERATE = 30
SKIP_FRAMES = 2
INPUT_SIZE = 224  # detector input


def run_single_trial(trial_num, num_frames, camera, detector, headless=False):
    """
    Run one trial: capture + detect for `num_frames` frames.
    Returns dict of per-frame metrics.
    """
    print(f"\n--- Trial {trial_num} ---")

    # Warmup: skip first 10 frames (camera auto-exposure)
    for _ in range(10):
        camera.capture_frame()
        time.sleep(0.02)

    frame_metrics = []
    detections_total = 0

    for i in range(num_frames):
        # 1. Capture
        t_capture_start = time.perf_counter()
        frame = camera.capture_frame()
        if frame is None:
            continue
        t_capture = time.perf_counter() - t_capture_start

        # 2. Detect (includes preprocess + invoke + NMS)
        t_detect_start = time.perf_counter()
        detections = detector.detect(frame)
        t_detect = time.perf_counter() - t_detect_start

        # 3. Draw (rendering cost)
        t_draw_start = time.perf_counter()
        annotated = detector.draw_detections(frame, detections)
        t_draw = time.perf_counter() - t_draw_start

        # Total pipeline time for this frame
        t_total = t_capture + t_detect + t_draw

        # Record
        det_count = len(detections)
        max_conf = max((d[4] for d in detections), default=0.0)
        detections_total += det_count

        frame_metrics.append({
            "frame": i + 1,
            "capture_ms": t_capture * 1000,
            "inference_ms": t_detect * 1000,
            "draw_ms": t_draw * 1000,
            "total_ms": t_total * 1000,
            "pipeline_fps": 1.0 / t_total if t_total > 0 else 0,
            "detections": det_count,
            "max_confidence": round(max_conf, 3),
        })

        # Optional display
        if not headless:
            try:
                disp = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                cv2.putText(disp, f"Trial {trial_num} | Frame {i+1}/{num_frames}",
                            (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.imshow("Benchmark", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                pass

    # Summarise this trial
    if not frame_metrics:
        return None

    inf_times = [m["inference_ms"] for m in frame_metrics]
    fps_vals = [m["pipeline_fps"] for m in frame_metrics]
    confs = [m["max_confidence"] for m in frame_metrics if m["max_confidence"] > 0]

    summary = {
        "trial": trial_num,
        "frames": len(frame_metrics),
        "inference_mean_ms": round(np.mean(inf_times), 2),
        "inference_std_ms": round(np.std(inf_times), 2),
        "inference_min_ms": round(np.min(inf_times), 2),
        "inference_max_ms": round(np.max(inf_times), 2),
        "pipeline_fps_mean": round(np.mean(fps_vals), 2),
        "pipeline_fps_std": round(np.std(fps_vals), 2),
        "total_detections": detections_total,
        "detection_rate": round(sum(1 for m in frame_metrics if m["detections"] > 0) / len(frame_metrics), 3),
        "avg_confidence": round(np.mean(confs), 3) if confs else 0,
        "per_frame": frame_metrics,
    }

    print(f"  Inference: {summary['inference_mean_ms']:.1f} ± {summary['inference_std_ms']:.1f} ms")
    print(f"  Pipeline FPS: {summary['pipeline_fps_mean']:.1f} ± {summary['pipeline_fps_std']:.1f}")
    print(f"  Detection rate: {summary['detection_rate']*100:.1f}%  (avg conf {summary['avg_confidence']:.2f})")

    return summary


def main():
    parser = argparse.ArgumentParser(description="YOLO Detection Benchmark")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials (default: 10)")
    parser.add_argument("--frames", type=int, default=100, help="Frames per trial (default: 100)")
    parser.add_argument("--headless", action="store_true", help="No display window")
    args = parser.parse_args()

    print("=" * 60)
    print("  YOLO Detection Benchmark")
    print(f"  Model: {MODEL_PATH.name}")
    print(f"  Trials: {args.trials}  |  Frames/trial: {args.frames}")
    print(f"  Camera: {CAMERA_RESOLUTION}  |  Skip: {SKIP_FRAMES}")
    print(f"  Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    print("=" * 60)

    # ---- Init camera ----
    camera = CameraCapture(resolution=CAMERA_RESOLUTION, framerate=CAMERA_FRAMERATE)
    if not camera.initialize():
        print("ERROR: Camera init failed")
        sys.exit(1)

    # ---- Init detector ----
    detector = TFLiteDetector(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
    )
    if not detector.load_model():
        print("ERROR: Model load failed")
        sys.exit(1)

    # ---- Run trials ----
    trial_summaries = []
    for t in range(1, args.trials + 1):
        summary = run_single_trial(t, args.frames, camera, detector, headless=args.headless)
        if summary:
            trial_summaries.append(summary)
        # Brief pause between trials
        time.sleep(1)

    camera.release()
    cv2.destroyAllWindows()

    if not trial_summaries:
        print("No successful trials.")
        sys.exit(1)

    # ---- Aggregate across trials ----
    inf_means = [s["inference_mean_ms"] for s in trial_summaries]
    fps_means = [s["pipeline_fps_mean"] for s in trial_summaries]
    det_rates = [s["detection_rate"] for s in trial_summaries]
    avg_confs = [s["avg_confidence"] for s in trial_summaries if s["avg_confidence"] > 0]

    print("\n" + "=" * 60)
    print("  AGGREGATE RESULTS  (N={} trials, {} frames each)".format(len(trial_summaries), args.frames))
    print("=" * 60)
    print(f"  Inference:      {np.mean(inf_means):.1f} ± {np.std(inf_means):.1f} ms")
    print(f"  Pipeline FPS:   {np.mean(fps_means):.1f} ± {np.std(fps_means):.1f}")
    print(f"  Detection rate: {np.mean(det_rates)*100:.1f} ± {np.std(det_rates)*100:.1f} %")
    if avg_confs:
        print(f"  Avg confidence: {np.mean(avg_confs):.3f} ± {np.std(avg_confs):.3f}")
    print("=" * 60)

    # ---- Save results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full JSON
    results = {
        "timestamp": timestamp,
        "config": {
            "model": MODEL_PATH.name,
            "input_size": INPUT_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "camera_resolution": list(CAMERA_RESOLUTION),
            "skip_frames": SKIP_FRAMES,
            "trials": args.trials,
            "frames_per_trial": args.frames,
        },
        "aggregate": {
            "inference_mean_ms": round(float(np.mean(inf_means)), 2),
            "inference_std_ms": round(float(np.std(inf_means)), 2),
            "pipeline_fps_mean": round(float(np.mean(fps_means)), 2),
            "pipeline_fps_std": round(float(np.std(fps_means)), 2),
            "detection_rate_mean": round(float(np.mean(det_rates)), 3),
            "detection_rate_std": round(float(np.std(det_rates)), 3),
        },
        "trials": trial_summaries,
    }

    json_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        # Remove per-frame data from JSON to keep file reasonable
        slim = dict(results)
        slim["trials"] = [{k: v for k, v in t.items() if k != "per_frame"} for t in trial_summaries]
        json.dump(slim, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # CSV of per-frame data (every frame from every trial)
    csv_path = RESULTS_DIR / f"per_frame_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trial", "frame", "capture_ms", "inference_ms", "draw_ms",
            "total_ms", "pipeline_fps", "detections", "max_confidence"
        ])
        writer.writeheader()
        for s in trial_summaries:
            for fm in s["per_frame"]:
                row = {"trial": s["trial"], **fm}
                writer.writerow(row)
    print(f"Per-frame CSV: {csv_path}")

    # Print table for paper
    print("\n--- Copy-paste for paper (LaTeX table row) ---")
    print(f"YOLOv8n FP16 & {INPUT_SIZE}×{INPUT_SIZE} & "
          f"{np.mean(inf_means):.1f} ± {np.std(inf_means):.1f} & "
          f"{np.mean(fps_means):.1f} ± {np.std(fps_means):.1f} & "
          f"{np.mean(det_rates)*100:.1f} \\\\")


if __name__ == "__main__":
    main()
