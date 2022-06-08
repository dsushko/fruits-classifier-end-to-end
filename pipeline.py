from colorsys import rgb_to_yiq
import os
from tkinter import image_names

import cv2

from model.config import ModelConfig
import numpy as np
from sklearn.metrics import accuracy_score
from utils.utils import EligibleModules
from model.preprocessing.preprocessor import FruitsPreprocessor
from utils.validator import ClassifierValidator
from utils.utils import load_cfg
from utils.logger import get_logger

class ModelRunner:

    def __init__(self, cfg_path, data_folder, validate=True):
        self._cfg = ModelConfig.parse_obj(load_cfg(cfg_path)).dict()
        self._logger = get_logger('fruits-classifier')

        self.data_folder = data_folder
        self.validate = validate

        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None


    def process_data_pair_for_classification(self, X, y, preprocessor):
        # check whether we can miss encoding
        X = preprocessor.transform(X)
        return X, y


    def read_imgs_and_make_labels(self, directory, preprocessor):

        images_count = 0
        for (dirpath, dirnames, filenames) in os.walk(directory):
            images_count += len([os.path. join(dirpath, file) for file in filenames])
        
        X = np.empty((images_count, preprocessor.resize_value, preprocessor.resize_value, 3))
        y = np.empty((images_count,), dtype=object)
        curr_img_ind = 0
        for label in os.listdir(directory):
            for pic in os.listdir(directory + label):
                try:
                    curr_img_path = directory + label + '/' + pic
                    stream = open(curr_img_path, "rb")
                    bytes = bytearray(stream.read())
                    numpyarray = np.asarray(bytes, dtype=np.uint8)
                    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
                    rgbImage = cv2.cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
                    X[curr_img_ind,:,:,:] = preprocessor.unificate_one(rgbImage)
                    y[curr_img_ind] = label
                except BaseException as err:
                   self._logger.error(f'Error while processing file {curr_img_path}: {err}')
                   pass
        return X, y


    def prepare_train_test(self, train_dir, validation_dir, preprocessor):
        self._logger.info('Reading train and test data...')
        train_X, train_y = self.read_imgs_and_make_labels(train_dir, preprocessor)
        test_X, test_y = self.read_imgs_and_make_labels(validation_dir, preprocessor)

        self._logger.info('Processing train and test data...')
        train_X, train_y = \
            self.process_data_pair_for_classification(train_X, train_y, preprocessor)
        test_X, test_y = \
            self.process_data_pair_for_classification(test_X, test_y, preprocessor)

        self._logger.info('Processing train and test data finished')

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y


    def run(self):
        self._logger.info('Pipeline start')

        train_dir = self.data_folder + '/train/'
        validation_dir = self.data_folder + '/test/'

        prepr_cfg = self._cfg['preprocessing']
        preprocessor = FruitsPreprocessor(prepr_cfg)
        self._logger.debug(f'Preprocessor\'s params:{prepr_cfg["params"]}')
        self._logger.debug(f'Preprocessor\'s unification steps:{prepr_cfg["unification_steps"]}')
        self._logger.debug(f'Preprocessor\'s processing steps:{prepr_cfg["processing_steps"]}')
        self.prepare_train_test(train_dir, validation_dir, preprocessor=preprocessor)

        classifier_name = self._cfg['classifier']['name']
        classifier_params = self._cfg['classifier']['params'] or {}

        self._logger.info(f'Initializing {classifier_name} classifier...')

        classifier = EligibleModules().classifiers[classifier_name](**classifier_params)
        print(self.train_X.shape, self.train_y.shape)
        self._logger.info('Fitting classifier...')
        classifier.fit(self.train_X, self.train_y)
        self._logger.info('Predicting...')
        pred_y = classifier.predict(self.test_X)

        if self.validate:
            self._logger.info('Validate option was chosen, validating model...')
            validator = ClassifierValidator()
            self._logger.info('Validating...')
            val_stats = \
                validator.validate_preds(pred_y, self.test_y, metrics=[accuracy_score])
            self._logger.info(f'Validation results: {val_stats}')
            self._logger.debug(f'Confusion matrix:')
            self._logger.debug(validator.confusion_matrix)
        self._logger('Done.')
