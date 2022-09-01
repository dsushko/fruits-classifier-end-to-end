from skimage.segmentation import mark_boundaries
import os
import numpy as np
import cv2
from skimage import io

from lime import lime_image
from explainability.config import LimeExplainabilityConfig
from utils.globalparams import GlobalParams
from utils.utils import read_rgb_image

def label_to_num(labels_arr, input):
    return {labels_arr[i]:i for i in range(len(labels_arr))}[input]
    
class LimeExplainer:

    def __init__(self, **kwargs):
        for param, val in LimeExplainabilityConfig.parse_obj(kwargs):
            setattr(self, param, val)
        self.explainer = lime_image.LimeImageExplainer()

    def make_explainability_example(self, pic_path, preprocessor, trained_model):
        img_to_explain = read_rgb_image(pic_path)
        img_to_explain = preprocessor.preprocess_one(img_to_explain)
        pred = label_to_num(
            trained_model.labels,
            trained_model.predict(img_to_explain[None, ...])[0]
        )
        explanation = self.explainer.explain_instance(
            img_to_explain, 
            trained_model.get_prob_matrix,
            labels=range(len(trained_model.labels))
        )
        explained_img, mask = explanation.get_image_and_mask(
            pred, 
            positive_only=self.positive_only,
            hide_rest=self.hide_rest,
            num_features=self.num_features,
            min_weight=self.min_weight
        )
        img_filename = os.path.basename(pic_path)
        img_class = pic_path.replace('/', '\\').split('\\')[-2]
        io.imsave(
            os.path.join(
                GlobalParams().explainability_results, 
                f'ithink_its_{trained_model.labels[pred]}_and_its_{img_class}_explained_{img_filename}'
            ),
            mark_boundaries(explained_img, mask)
        )

        io.imsave(
            os.path.join(
                GlobalParams().explainability_results, 
                f'ithink_its_{trained_model.labels[pred]}_and_its_{img_class}_orig_{img_filename}'
            ),
            img_to_explain, 
        )
