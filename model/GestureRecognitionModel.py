import pandas as pd 
import numpy as np 
from xgboost import XGBClassifier


class GestureRecognitionModel():
    def __init__(self, model_name="xgboost"):
        self.model_name = model_name
        self.model = self.get_model()

    def train(self, x, y):
        self.model.fit(x, y)

    def evaluate(self, x, y):
        return self.model.score(x, y)
        

    def predict(self, x):
        return self.model.predict(x)
        

    def get_model(self):
        if self.model_name == "xgboost":
            return XGBClassifier()
        else:
            raise ValueError(f"Model - {self.model_name} not implemented.")
    
    
        