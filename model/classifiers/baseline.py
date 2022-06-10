from sklearn import svm
from utils.eligiblemodules import EligibleModules

@EligibleModules.register_classifier
class BaselineClassifier:

    def __init__(self, **kwargs):
        self.model = svm.SVC(**kwargs)
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
