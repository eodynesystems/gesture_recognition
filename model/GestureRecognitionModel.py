import pandas as pd 
import numpy as np 
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

class GestureRecognitionModel():
    def __init__(self, model_name="xgboost", class_weight = 'None'):
        self.model_name = model_name
        self.model = self.get_model()
        if class_weight:
            self.class_weight = class_weight

    def train(self, x, y):
        self.model.fit(x, y)

    def evaluate(self, x, y):
        return self.model.score(x, y)
        
    def predict(self, x):
        return self.model.predict(x)
        
    def get_model(self):
        if self.model_name == "xgboost":
            return XGBClassifier()
        elif self.model_name.startswith("svm"):
            return SVC(kernel=self.model_name.split("_")[-1], class_weight = self.class_weight)
        elif self.model_name == "mlp":
            return MLPClassifier(random_state=1,hidden_layer_sizes = 50,max_iter=500)
        else:
            raise ValueError(f"Model - {self.model_name} not implemented.")
