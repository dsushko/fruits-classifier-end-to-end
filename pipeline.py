import os

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from utils.utils import EligibleModules
from preprocessor import FruitsPreprocessor
from utils.validator import ClassifierValidator
from utils.utils import load_cfg
from utils.logger import get_logger

class ModelRunner:

    def __init__(self, cfg_path, data_folder, validate=True):
        self._cfg = load_cfg(cfg_path)
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
        #self.labels_encoder = LabelEncoder()
        #y = self.labels_encoder.fit_transform(y)
        return X, y


    def read_imgs_and_make_labels(self, directory):
        # ???? investigate if it's the best solution
        X = []
        y = []
        for label in os.listdir(directory):
            for pic in os.listdir(directory + label):
                try:
                    curr_img_path = directory + label + '/' + pic
                    stream = open(curr_img_path, "rb")
                    bytes = bytearray(stream.read())
                    numpyarray = np.asarray(bytes, dtype=np.uint8)
                    bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

                    X.append(cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB))
                    y.append(label)
                except BaseException as err:
                    self._logger.error(f'Couldn\'t open file {curr_img_path}: {err}')
                    pass
        return X, y


    def prepare_train_test(self, train_dir, validation_dir, preprocessor):
        self._logger.info('Reading train and test data...')
        train_X, train_y = self.read_imgs_and_make_labels(train_dir)
        test_X, test_y = self.read_imgs_and_make_labels(validation_dir)

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

        # warning: path + str = ?
        train_dir = self.data_folder + '/train/'
        validation_dir = self.data_folder + '/test/'

        prepr_cfg = self._cfg['preprocessing']
        preprocessor = FruitsPreprocessor(prepr_cfg)
        self._logger.debug(f'Preprocessor\'s params:{prepr_cfg["ctor_params"]}')
        self._logger.debug(f'Preprocessor\'s steps:{prepr_cfg["steps"]}')
        self.prepare_train_test(train_dir, validation_dir, preprocessor=preprocessor)

        classifier_name = self._cfg['classifier']['name']
        classifier_params = self._cfg['classifier']['params'] or {}

        self._logger.info(f'Initializing {classifier_name} classifier...')

        classifier = EligibleModules().classifiers[classifier_name](**classifier_params)

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
