import os

class GlobalParams:
    __instance = None
    num_classes: int
    build: str
    data_path: str = './data/'
    models_cfg_path: str = './cfg/model/'
    explainability_results: str = './explainability_demo/'

    @staticmethod 
    def count_classes():
        return len(os.listdir(GlobalParams.data_path + 'train/'))

    @staticmethod
    def build_cfg_path():
        return GlobalParams.models_cfg_path + GlobalParams.build + '.yaml'

    def __new__(cls):
        if GlobalParams.__instance is None:
            GlobalParams.__instance = object.__new__(cls)
            GlobalParams.num_classes = GlobalParams.count_classes()
            GlobalParams
        return GlobalParams.__instance

    @staticmethod
    def update_params(kwargs: dict):
        for arg, val in kwargs.items():
            setattr(GlobalParams, arg, val)