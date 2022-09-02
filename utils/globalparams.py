import os

class GlobalParams:
    __instance = None
    num_classes: int
    build: str
    data_path: str = './data/'
    models_cfg_path: str = './cfg/model/'
    tuning_cfg_path: str = './cfg/tuning/'
    tuning_results_path: str = './data/tuning_results/'
    explainability_results: str = './explainability_demo/'
    saved_models_folder: str = 'saved_models/'

    @staticmethod 
    def count_classes():
        return len(os.listdir(GlobalParams.data_path + 'train/'))

    @staticmethod
    def build_cfg_path():
        return GlobalParams.models_cfg_path + GlobalParams.build + '.yaml'

    #@staticmethod
    #def build_local_model_path():
    #    return os.path.join(GlobalParams().data_path, '/saved_models/', build, build + 'json')

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