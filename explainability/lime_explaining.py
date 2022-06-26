from skimage.segmentation import mark_boundaries
import os
import numpy as np
import cv2
from skimage import io

from lime import lime_image
from explainability.config import LimeExplainabilityConfig
from utils.globalparams import GlobalParams
from utils.utils import read_rgb_image

class LimeExplainer:

    def __init__(self, **kwargs):
        for param, val in LimeExplainabilityConfig.parse_obj(kwargs):
            setattr(self, param, val)
        self.explainer = lime_image.LimeImageExplainer()
        
    def make_explainability_expamples(self, preprocessor, trained_model):
        test_picc_path = os.path.join(GlobalParams().data_path, 'test')
        for pic_label in os.listdir(test_picc_path):
            random_pic_name = np.random.choice(os.listdir(os.path.join(test_picc_path, pic_label)))
            random_pic_path = os.path.join(test_picc_path, pic_label, random_pic_name)
            random_img = read_rgb_image(random_pic_path)
            random_img = preprocessor.preprocess_one(random_img)
            pred = trained_model.predict(random_img[None, ...])[0]
            explanation = self.explainer.explain_instance(
                random_img, 
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

            io.imsave(
                os.path.join(
                    GlobalParams().explainability_results, 
                    'explained_' + pic_label + '_' + trained_model.labels[pred] + '_' + random_pic_name
                ),
                mark_boundaries(explained_img, mask)
            )

            io.imsave(
                os.path.join(
                    GlobalParams().explainability_results, 
                    'explained_' + pic_label  + '_' + trained_model.labels[pred] + '_' + '_ORIG_' + random_pic_name
                ),
                random_img, 
            )
