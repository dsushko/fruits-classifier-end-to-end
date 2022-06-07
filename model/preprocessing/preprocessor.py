import cv2
import numpy as np

class FruitsPreprocessor:

    def __init__(self, cfg):
        params = cfg['params'] or {}
        for param, value in params.items():
            setattr(self, param, value)
        self.steps = cfg['steps']
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
        for preprocess_step in self.steps:
            img = getattr(self, preprocess_step)(img)
        return img

    def flatten(self, img):
        return img.flatten()

    def transform(self, imgs):
        return np.array([self.preprocess_one(img) for img in imgs])