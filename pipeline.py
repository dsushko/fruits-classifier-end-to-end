import logging
import os

import cv2

from model.config import ModelConfig
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from utils.eligiblemodules import EligibleModules
from utils.globalparams import GlobalParams
from model.preprocessing.preprocessor import FruitsPreprocessor
from utils.validator import ClassifierValidator
from utils.utils import load_cfg
from utils.logging import init_logger
from explainability.lime_explaining import LimeExplainer

init_logger(__name__)
logger = logging.getLogger(__name__)

class ModelRunner:

    def __init__(self, build, validate_flag=True):

        self._cfg = ModelConfig.parse_obj(
            load_cfg(GlobalParams().model_cfg_path)
        ).dict()

        self.data_folder = GlobalParams().data_folder
        self.validate_flag = validate_flag

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
                    curr_img_ind += 1
                except BaseException as err:
                   logger.error(f'Error while processing file {curr_img_path}: {err}')
                   pass
        return X[:curr_img_ind,:,:,:], y[:curr_img_ind]


    def prepare_train_test(self, train_dir, validation_dir, preprocessor):
        logger.info('Reading train and test data...')
        train_X, train_y = self.read_imgs_and_make_labels(train_dir, preprocessor)
        test_X, test_y = self.read_imgs_and_make_labels(validation_dir, preprocessor)

        logger.info('Processing train and test data...')
        train_X, train_y = \
            self.process_data_pair_for_classification(train_X, train_y, preprocessor)
        test_X, test_y = \
            self.process_data_pair_for_classification(test_X, test_y, preprocessor)

        logger.info('Processing train and test data finished')

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y


    def run(self):
        logger.info('Pipeline start')

        train_dir = self.data_folder + '/train/'
        validation_dir = self.data_folder + '/test/'

        prepr_cfg = self._cfg['preprocessing']
        preprocessor = FruitsPreprocessor(prepr_cfg)
        logger.debug(f'Preprocessor\'s params:{prepr_cfg["params"]}')
        logger.debug(f'Preprocessor\'s unification steps:{prepr_cfg["unification_steps"]}')
        logger.debug(f'Preprocessor\'s processing steps:{prepr_cfg["processing_steps"]}')
        self.prepare_train_test(train_dir, validation_dir, preprocessor=preprocessor)



        classifier_name = self._cfg['classifier']['name']
        classifier_params = self._cfg['classifier']['params']

        logger.info(f'Initializing {classifier_name} classifier...')

        classifier = EligibleModules().classifiers[classifier_name](**classifier_params)
        logger.info('Fitting classifier...')
        classifier.fit(self.train_X, self.train_y)
        logger.info('Predicting...')
        pred_y = classifier.predict(self.test_X)


        if self.validate_flag:
            logger.info('Validate option was chosen, validating model...')
            validator = ClassifierValidator()
            logger.info('Validating...')
            val_stats = \
                validator.validate_preds(pred_y, self.test_y, metrics=[accuracy_score])
            logger.info(f'Validation results: {val_stats}')
            logger.debug(f'Confusion matrix:')
            for matrix_ind in range(len(validator.confusion_matrix)):
                logger.debug(f'{validator.cf_labels[matrix_ind]}: {list(validator.confusion_matrix[matrix_ind])}')

        logger.info('Creating explainability examples')
        explainer = LimeExplainer()

        explainer.make_explainability_expamples(preprocessor, classifier)

        logger.info('Done.')
