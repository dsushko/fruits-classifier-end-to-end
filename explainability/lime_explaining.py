from skimage.segmentation import mark_boundaries
import os
import numpy as np
import cv2

from lime import lime_image
from explainability.config import LimeExplainabilityConfig
from utils.globalparams import GlobalParams

class LimeExplainer:

    def __init__(self, **kwargs):
        for param, val in LimeExplainabilityConfig.parse_obj(kwargs):
            setattr(self, param, val)
        self.explainer = lime_image.LimeImageExplainer()
        
    def make_explainability_expamples(self, preprocessor, trained_model):
        data_path = GlobalParams().data_path
        for pic_label in os.listdir(data_path):
            random_pic_name = np.random.choice(os.listdir(os.path.join(data_path, pic_label)))
            random_pic_path = os.path.join(data_path, pic_label, random_pic_name)
            image = cv2.cvtColor(
                cv2.imread(random_pic_path),
                cv2.COLOR_BGR2RGB
            )
            image = preprocessor.preprocess_one(image)
            pred = trained_model.predict(image[None, ...])
            explanation = self.explainer.explain_instance(
                image, 
                trained_model, 
                labels=[pred,], 
                top_labels=None
            )
            img, mask = explanation.get_image_and_mask(
                pred, 
                positive_only=self.positive_only,
                hide_rest=self.hide_rest,
                num_features=self.num_features,
                min_weight=self.min_weight
            )
            cv2.imwrite(
                os.path.join(
                    GlobalParams().explainability_results, 
                    'explained_' + random_pic_name
                ),
                mark_boundaries(img, mask)
            )
