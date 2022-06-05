import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

class ClassifierValidator:

    def __init__(self):
        pass

    def validate_model(self, model, test_df, test_y, 
                 train_df=None, 
                 train_y=None,  
                 predict_only=True,
                 metrics=[]
                ):
        if not predict_only:
            model.fit(train_df, train_y)

        pred_y = model.predict(test_df)

        self.cf_labels = np.sort(np.unique(test_y))
        self.confusion_matrix = confusion_matrix(test_y, pred_y, labels=self.cf_labels)
        
        scores = {}
        for metric in metrics:
            scores[metric.__name__] = metric(test_y, pred_y)
        
        return scores
    
    def validate_preds(self, pred_y, true_y, metrics=[]):

        self.cf_labels = np.sort(np.unique(true_y))
        self.confusion_matrix = confusion_matrix(true_y, pred_y, labels=self.cf_labels)
        
        scores = {}
        for metric in metrics:
            scores[metric.__name__] = metric(true_y, pred_y)
        
        return scores
