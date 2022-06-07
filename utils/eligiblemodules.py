class EligibleModules:
    """Utility for components registration."""
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