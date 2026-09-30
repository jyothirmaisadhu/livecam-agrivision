"""Step 6: desktop live-camera demo. Runs the same pipeline the phone app will run:

frame -> (every Nth frame) disease + pest + damage models -> smoothing -> decision engine
      -> overlay on video + spoken advice.

Any model can be omitted; the demo runs with whatever you have trained so far.
Example:
  python -m agrivision.pipeline.live --source 0 \
      --disease models/disease/disease_mobilenetv3.tflite --labels models/disease/labels.json \
      --pest models/pest/pest_yolo11n.pt --damage models/damage/damage_unet.tflite --voice
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from agrivision.config import load_settings
from agrivision.damage.severity import severity_from_mask
from agrivision.decision.engine import build_report
from agrivision.pipeline.models import DamageSegmenter, DiseaseClassifier, PestDetector
from agrivision.pipeline.overlay import draw
from agrivision.pipeline.smoothing import MajorityVote
from agrivision.pipeline.voice import Speaker


class LivePipeline:
    def __init__(
        self, disease: DiseaseClassifier | None, pest: PestDetector | None, damage: DamageSegmenter | None
    ):
        self.settings = load_settings().live
        self.disease, self.pest, self.damage = disease, pest, damage
        self.vote = MajorityVote(self.settings.smoothing_window)
        self.frame_index = 0
        self.last = (None, [], None)

    def process(self, frame_bgr):
        """Returns (report, pests, mask) — recomputed every N frames, cached in between."""
        self.frame_index += 1
        if (self.frame_index - 1) % self.settings.process_every_n_frames:
            return self.last
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        disease = None
        if self.disease:
            raw = self.disease.predict(rgb)
            stable = self.vote.update(
                raw.label if raw.confidence >= self.settings.disease_confidence else None
            )
            disease = raw if stable == raw.label else None

        pests = self.pest.predict(rgb) if self.pest else []
        mask, sev = None, None
        if self.damage and disease and not disease.label.endswith("_healthy"):
            mask = self.damage.predict(rgb)
            sev = severity_from_mask(mask)

        report = build_report(disease, pests, sev.percent if sev else None, sev.level if sev else None)
        self.last = (report, pests, mask)
        return self.last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam index or video/image path")
    ap.add_argument("--disease", type=Path)
    ap.add_argument("--labels", type=Path, default=Path("models/disease/labels.json"))
    ap.add_argument("--pest", type=Path, help=".pt or .tflite YOLO weights")
    ap.add_argument("--damage", type=Path)
    ap.add_argument("--voice", action="store_true")
    ap.add_argument("--save", type=Path, help="write annotated video here")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    live = load_settings().live
    pipeline = LivePipeline(
        DiseaseClassifier(args.disease, args.labels) if args.disease else None,
        PestDetector(args.pest, live.pest_confidence) if args.pest else None,
        DamageSegmenter(args.damage) if args.damage else None,
    )
    speaker = Speaker(args.voice, live.speak_cooldown_seconds)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        report, pests, mask = pipeline.process(frame)
        annotated = draw(frame, report, pests, mask)
        if report:
            speaker.say(report.spoken)
        if args.save:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(annotated)
        if not args.no_display:
            cv2.imshow("LiveCam AgriVision (q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    if report:
        print(report.spoken)


if __name__ == "__main__":
    main()
