import cv2
import json
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import os

# ── Load params from DVC params.yaml ──────────────────────────────────────────
def load_params():
    params_path = Path("params.yaml")
    if params_path.exists():
        with open(params_path) as f:
            return yaml.safe_load(f)
    # Fallback defaults if params.yaml not found
    return {
        "detect": {
            "conf_threshold": 0.5,
            "iou_threshold":  0.45,
            "img_size":       640,
            "model_path":     r'runs/detect/ppe_detector/weights/best (1).pt',
            "sources":        ["indianworkers.mp4", "JapanPPE.mp4"]
        }
    }

# ── Class config: class_name → (color BGR, is_violation) ──────────────────────
CLASS_CONFIG = {
    'Hardhat':        ((0, 255, 0),   False),
    'NO-Hardhat':     ((0, 0, 255),   True),
    'Safety Vest':    ((0, 255, 0),   False),
    'NO-Safety Vest': ((0, 0, 255),   True),
    'Mask':           ((0, 255, 0),   False),
    'NO-Mask':        ((0, 0, 255),   True),
    'Person':         ((255, 255, 0), False),
    'Safety Cone':    ((255, 165, 0), False),
    'machinery':      ((255, 0, 255), False),
    'vehicle':        ((255, 0, 0),   False),
}

# ── Detect if running in CI (GitHub Actions sets CI=true automatically) ────────
IS_CI = os.environ.get('CI', 'false').lower() == 'true'


class PPEDetector:
    def __init__(self, model_path=None, conf=0.5, iou=0.45, img_size=640):
        # Load params.yaml; constructor args override params if explicitly passed
        self.params = load_params()["detect"]

        self.model_path = model_path or self.params.get("model_path")
        self.conf       = conf      or self.params.get("conf_threshold", 0.5)
        self.iou        = iou       or self.params.get("iou_threshold",  0.45)
        self.img_size   = img_size  or self.params.get("img_size", 640)

        self.model         = YOLO(self.model_path)
        self.violation_log = []

        # Tracks per-class detection counts for metrics
        self.class_counts  = {cls: 0 for cls in CLASS_CONFIG}

    # ── Core detection & drawing ───────────────────────────────────────────────
    def draw_detections(self, frame, results):
        violations = []

        for result in results:
            boxes = result.boxes  # [x, y, w, h, obj, p1, p2, ...]
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = float(box.conf[0])
                cls_id     = int(box.cls[0])
                cls_name   = self.model.names[cls_id]

                color, is_violation = CLASS_CONFIG.get(cls_name, ((255, 255, 0), False))
                label = f"{cls_name} {conf_score:.2f}"

                # Track class counts for metrics
                if cls_name in self.class_counts:
                    self.class_counts[cls_name] += 1

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Label background + text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if is_violation:
                    violations.append(cls_name)

        # Violation alert banner
        if violations:
            alert_text = f"VIOLATION: {', '.join(set(violations))}"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 200), -1)
            cv2.putText(frame, alert_text, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            self.violation_log.append({
                'time':       datetime.now().strftime('%H:%M:%S'),
                'violations': list(set(violations))
            })

        return frame, violations

    # ── Save DVC metrics ───────────────────────────────────────────────────────
    def save_metrics(self, source_tag="video", extra: dict = None):
        """
        Runs model.val() on dataset.yaml and writes metrics/eval_results.json
        so DVC can track & compare across runs.
        """
        Path("metrics").mkdir(exist_ok=True)

        try:
            val_results = self.model.val(data="dataset.yaml")
            metrics = {
                "source":          source_tag,
                "timestamp":       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "mAP50":           round(float(val_results.box.map50),  4),
                "mAP50-95":        round(float(val_results.box.map),    4),
                "precision":       round(float(val_results.box.p.mean()), 4),
                "recall":          round(float(val_results.box.r.mean()), 4),
                "violation_events": len(self.violation_log),
                "class_counts":    {k: v for k, v in self.class_counts.items() if v > 0},
            }
        except Exception as e:
            print(f"⚠️  Could not run validation (no dataset.yaml?): {e}")
            # Still save what we know from the detection run
            metrics = {
                "source":          source_tag,
                "timestamp":       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "violation_events": len(self.violation_log),
                "class_counts":    {k: v for k, v in self.class_counts.items() if v > 0},
            }

        if extra:
            metrics.update(extra)

        with open("metrics/eval_results.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n📊 Metrics saved → metrics/eval_results.json")
        if "mAP50" in metrics:
            print(f"   mAP50:      {metrics['mAP50']}")
            print(f"   Precision:  {metrics['precision']}")
            print(f"   Recall:     {metrics['recall']}")
        print(f"   Violations: {metrics['violation_events']} events")

    # ── Image detection ────────────────────────────────────────────────────────
    def detect_image(self, image_path, output_path='output.jpg'):
        """Detect PPE in a single image and save annotated result."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        results = self.model(frame, conf=self.conf, iou=self.iou, imgsz=self.img_size)
        annotated, violations = self.draw_detections(frame, results)
        cv2.imwrite(output_path, annotated)
        print(f"✅ Saved → {output_path} | Violations: {violations}")

        self.save_metrics(source_tag=image_path)
        return annotated, violations

    # ── Video / webcam detection ───────────────────────────────────────────────
    def detect_video(self, source=0, output_path='output.mp4'):
        """Detect PPE in a video file or live webcam feed."""
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        out = None
        if isinstance(source, str):  # save output only for video files
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"🎬 Processing: {source}  →  {output_path}")

        if not IS_CI:
            print("Press 'q' to quit")

        frame_count      = 0
        violation_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results   = self.model(frame, conf=self.conf, iou=self.iou, imgsz=self.img_size)
            annotated, violations = self.draw_detections(frame, results)

            frame_count += 1
            if violations:
                violation_frames += 1

            # ✅ Skip GUI display in CI — no screen available on runner
            if not IS_CI:
                cv2.imshow('PPE Detection — Press Q to quit', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if out is not None:
                out.write(annotated)

        cap.release()
        if out is not None:
            out.release()

        # ✅ Skip destroyAllWindows in CI — nothing to destroy
        if not IS_CI:
            cv2.destroyAllWindows()

        # ── Session summary ────────────────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"Session Summary — {source}")
        print(f"  Total frames:     {frame_count}")
        print(f"  Violation frames: {violation_frames}")
        print(f"  Violation events: {len(self.violation_log)}")
        for entry in self.violation_log:
            print(f"    [{entry['time']}] {', '.join(entry['violations'])}")
        print('='*50)

        self.save_metrics(
            source_tag=str(source),
            extra={"total_frames": frame_count, "violation_frames": violation_frames}
        )

        return self.violation_log

    # ── Batch: run all sources from params.yaml ────────────────────────────────
    def detect_all_sources(self):
        """Run detection on every source listed in params.yaml → detect.sources"""
        sources = self.params.get("sources", [])
        if not sources:
            print("⚠️  No sources found in params.yaml")
            return

        for source in sources:
            if not Path(source).exists():
                print(f"⚠️  Skipping {source} — file not found")
                continue

            suffix      = Path(source).suffix.lower()
            output_name = f"result/{Path(source).stem}_output{suffix}"
            Path("result").mkdir(exist_ok=True)

            print(f"\n{'='*50}")
            if suffix in ['.jpg', '.jpeg', '.png']:
                self.detect_image(source, output_path=output_name)
            else:
                self.detect_video(source, output_path=output_name)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = load_params()["detect"]

    detector = PPEDetector(
        model_path = p["model_path"],
        conf       = p["conf_threshold"],
        iou        = p["iou_threshold"],
        img_size   = p["img_size"],
    )

    # ── Choose your mode ───────────────────────────────────────────────────────

    # 1. Run ALL sources defined in params.yaml (recommended for DVC pipeline)
    detector.detect_all_sources()

    # 2. Webcam live detection
    # detector.detect_video(source=0)

    # 3. Single image
    # detector.detect_image('construction-safety.jpg')

    # 4. Single video
    # detector.detect_video('indianworkers.mp4')  # ← commented out to avoid double-run


'''
Notes:
  - detect_all_sources() is called by DVC via `dvc repro`
  - metrics are saved to metrics/eval_results.json after every run
  - params.yaml controls model path, confidence, sources — no hardcoding needed
  - result/ folder holds all annotated outputs
  - GUI display is skipped automatically in CI (CI=true env var)
'''