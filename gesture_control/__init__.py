"""Control Windows with hand gestures seen through a webcam."""

import os

# OpenCV's Media Foundation camera backend can take 30+ seconds to open a
# webcam while it probes hardware transforms. Without them it opens in about
# a second and still delivers full frame rate. Must be set before cv2 loads.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
# OpenCV warns about every camera number it probes that doesn't exist.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

__version__ = "1.0.0"
