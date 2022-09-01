import logging
import os

import cv2
import pickle

from model.config import ModelConfig

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from utils.eligiblemodules import EligibleModules
from utils.globalparams import GlobalParams
from model.preprocessing.preprocessor import FruitsPreprocessor
from utils.validator import ClassifierValidator
from utils.utils import load_cfg
from utils.logging import init_logger
from utils.utils import read_rgb_image
from explainability.lime_explaining import LimeExplainer

init_logger(__name__)
logger = logging.getLogger(__name__)

class ModelRunner:

    def __init__(self, mode, validate_flag=True, explainability_path=None):
        self.mode = mode
        self._cfg = ModelConfig.parse_obj(
            load_cfg(GlobalParams().build_cfg_path())
        ).dict()

        classifier_name = self._cfg['classifier']['name']
        classifier_params = self._cfg['classifier']['params']

        logger.info(f'Initializing {classifier_name}...')

        self.classifier = \
            EligibleModules().classifiers[classifier_name](**classifier_params)

        self.data_path = GlobalParams().data_path
        self.validate_flag = validate_flag
        self.img_path_to_explain = explainability_path

        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None


    def process_data_pair_for_classification(self, X, y, preprocessor):
        X = preprocessor.transform(X)
        if self._cfg['encode_labels']:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        return X, y

    def read_imgs_and_make_labels(self, directory, preprocessor, with_filenames=False):

        images_count = 0
        for (dirpath, dirnames, filenames) in os.walk(directory):
            images_count += len([os.path. join(dirpath, file) for file in filenames])
        
        X = np.empty((images_count, preprocessor.resize_value, preprocessor.resize_value, 3))
        y = np.empty((images_count,), dtype=object)
        if with_filenames:
            filenames = np.empty((images_count,), dtype=object)
        curr_img_ind = 0
        for label in os.listdir(directory):
            for pic in os.listdir(directory + label):
                try:
                    curr_img_path = directory + label + '/' + pic
                    rgb_image = read_rgb_image(curr_img_path)
                    X[curr_img_ind,:,:,:] = preprocessor.unificate_one(rgb_image)
                    y[curr_img_ind] = label
                    if with_filenames:
                        filenames[curr_img_ind] = curr_img_path
                    curr_img_ind += 1
                except BaseException as err:
                   logger.error(f'Error while processing file {curr_img_path}: {err}')
                   pass
        if with_filenames:
            return X[:curr_img_ind,:,:,:], y[:curr_img_ind], filenames[:curr_img_ind]
        return X[:curr_img_ind,:,:,:], y[:curr_img_ind]


    def prepare_data_from_folder(self, dir, preprocessor, with_filenames=False):
        logger.info('Reading data from folder...')

        if with_filenames:
            X, y, filenames = \
                self.read_imgs_and_make_labels(dir, preprocessor, with_filenames=True)
        else:
            X, y = self.read_imgs_and_make_labels(dir, preprocessor)

        logger.info('Processing data from folder...')
        X, y = self.process_data_pair_for_classification(X, y, preprocessor)

        logger.info('Processing folder data finished')

        if with_filenames:
            return X, y, filenames
        return X, y


    def train(self):
        logger.info('Fitting classifier...')
        self.classifier.fit(self.train_X, self.train_y)
        self.save_clf()

    def get_clf_files_path(self):
        classifier_name = GlobalParams().build
        data_storage = GlobalParams().data_path
        return os.path.join(data_storage, 'saved_models', classifier_name)
    
    def save_clf(self):
        return self.classifier.save_clf()

    def load_clf(self):
        return self.classifier.load_clf()

    def predict(self):
        logger.info('Predicting...')
        self.pred_y = self.classifier.predict(self.test_X)
        self.save_predictions()

    def save_predictions(self):
        pd.DataFrame([self.predict_filenames, self.pred_y]).T \
            .to_csv('preds.csv', index=False)

    def explainability(self, img_path):
        logger.info('Explaining given example')
        explainer = LimeExplainer()
        explainer.make_explainability_example(
            img_path,
            self.preprocessor, 
            self.classifier
        )

    def validate(self):
        validator = ClassifierValidator()
        logger.info('Validating...')
        val_stats = \
            validator.validate_preds(self.pred_y, self.test_y, metrics=[accuracy_score])
        logger.info(f'Validation results: {val_stats}')
        logger.debug(f'Confusion matrix:')
        for matrix_ind in range(len(validator.confusion_matrix)):
            logger.debug(f'{validator.cf_labels[matrix_ind]}: {list(validator.confusion_matrix[matrix_ind])}')

    def launch_selected_mode(self):
        if self.mode == 'train':
            train_dir = self.data_path + '/train/'

            self.train_X, self.train_y = \
                self.prepare_data_from_folder(train_dir, preprocessor=self.preprocessor)

            self.train()
        if self.mode == 'predict':
            validation_dir = self.data_path + '/test/'

            self.test_X, self.test_y, self.predict_filenames = \
                self.prepare_data_from_folder(
                    validation_dir, 
                    preprocessor=self.preprocessor,
                    with_filenames=True
                )

            self.classifier = self.load_clf()
            self.predict()

            if self.validate_flag:
                logger.info('Validate option was chosen, validating model...')
                self.validate()
        if self.mode == 'explainability':
            self.classifier = self.load_clf()
            self.explainability(self.img_path_to_explain)
        if self.mode == 'tuning':
            pass

    def run(self):
        logger.info('Pipeline start')

        prepr_cfg = self._cfg['preprocessing']
        self.preprocessor = FruitsPreprocessor(prepr_cfg)
        logger.debug(f'Preprocessor\'s params:{prepr_cfg["params"]}')
        logger.debug(f'Preprocessor\'s unification steps:{prepr_cfg["unification_steps"]}')
        logger.debug(f'Preprocessor\'s processing steps:{prepr_cfg["processing_steps"]}')

        self.launch_selected_mode()

        logger.info('Done.')
