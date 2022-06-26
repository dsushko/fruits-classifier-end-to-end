import yaml
import numpy as np
import cv2

def load_cfg(cfg_path):
    with open(cfg_path, 'r') as stream:
        return yaml.safe_load(stream)

def read_rgb_image(path):
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    bgr_image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image