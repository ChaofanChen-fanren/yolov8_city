# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import detect, segment, segmentcity
from .model import YOLO


__all__ = "segment", "detect", "YOLO", "segmentcity"
