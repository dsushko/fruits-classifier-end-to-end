import yaml

class EligibleModules:
    """Utility for registration of fast-track components."""
    __instance = None

    def __new__(cls, *_args, **_kwargs):
        if EligibleModules.__instance is None:
            EligibleModules.__instance = object.__new__(cls)
            EligibleModules.__instance.classifiers = {}

        return EligibleModules.__instance

    @classmethod
    def register_classifier(cls, classifier):
        """Registers a given classifier."""
        EligibleModules().classifiers.update({classifier.__name__: classifier})
        # TODO: import classifier
        return classifier


def load_cfg(cfg_path):
    with open(cfg_path, 'r') as stream:
        return yaml.safe_load(stream)