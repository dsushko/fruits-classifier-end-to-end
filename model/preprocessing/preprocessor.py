import cv2
import numpy as np

class FruitsPreprocessor:

    def __init__(self, cfg):
        params = cfg['params'] or {}
        for param, value in params.items():
            setattr(self, param, value)
        self.unification_steps = cfg['unification_steps']
        self.processing_steps = cfg['processing_steps']
        pass
    
    def resize(self, img):
        return cv2.resize(img, (self.resize_value, self.resize_value))

    def center_crop(self, img):
        width = img.shape[1]
        height = img.shape[0]

        center_x = int(width / 2)
        center_y = int(height / 2)

        min_axis_half = int(min(width, height) / 2)

        cropped = img[(center_y-min_axis_half):(center_y+min_axis_half),
                       center_x-min_axis_half:center_x+min_axis_half,:]
        return cropped

    def normalize(self, img):
        return cv2.normalize(img, None, 0, self.max_brightness, cv2.NORM_MINMAX)
    
    def preprocess_one(self, img):
        img = self.unificate_one(img)
        img = self.process_one(img)
        return img

    def process_one(self, img):
        for process_step in self.processing_steps:
            img = getattr(self, process_step)(img)
        return img

    def unificate_one(self, img):
        for unification_step in self.unification_steps:
            img = getattr(self, unification_step)(img)
        return img

    def flatten(self, img):
        return img.flatten()

    def transform(self, imgs):
        return np.array([self.process_one(img) for img in imgs])