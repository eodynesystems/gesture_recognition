# imports
import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from imblearn.under_sampling import RandomUnderSampler
from GestureRecognitionModel import GestureRecognitionModel

gestures_to_use = ["CloseHand", "Neutral", "OpenHand", 
                "WristSupination", "WristPronation",
                "ThumbAbduction", "ThumbAdduction",
                "Pinch", "Lateral", "Tripod", "Point"]   # "Pinch", "Lateral", "Tripod", "Point"
individual = False

def main():
    # initialise pdf 
    c = canvas.Canvas('../results_general.pdf', pagesize=letter)
    y = 2 * inch

    # load dataset 
    df = pd.read_csv("../data/features.csv")
    df = df[df.gesture.isin(gestures_to_use)]
    subjects = list(df.subject.unique())
    subjects.remove("Mario")
    subjects.remove("Hanaa")

    if not individual:
        subjects_to_remove = ["Mario", "Hanaa"]
        df_train = df[df["take"] != 1]
        df_train = df_train[~df_train.subject.isin(subjects_to_remove)]
        df_test = df[df["take"] == 1]
        
        gestures = df_train.gesture.unique()
        gesture_to_int = {gesture:i for i, gesture in enumerate(gestures)}
        int_to_gesture = {val:key for key, val in gesture_to_int.items()}
        features_to_keep = df_train.columns[:-4]
        X_train, y_train = df_train[features_to_keep], [gesture_to_int[gesture] for gesture in df_train.gesture]

        # train xgb model 
        model = GestureRecognitionModel("xgboost")
        model.train(X_train,y_train)

    # for each subject 
    for subject in tqdm(subjects):
        
        if individual:
            df_sub = df[df.subject == subject]
            # train test split 
            df_train = df_sub[df_sub["take"] != 1]
            df_test = df_sub[df_sub["take"] == 1]
            
            gestures = df_train.gesture.unique()
            gesture_to_int = {gesture:i for i, gesture in enumerate(gestures)}
            int_to_gesture = {val:key for key, val in gesture_to_int.items()}
            features_to_keep = df_train.columns[:-4]
            X_train, y_train = df_train[features_to_keep], [gesture_to_int[gesture] for gesture in df_train.gesture]
            X_test, y_test = df_test[features_to_keep], [gesture_to_int[gesture] for gesture in df_test.gesture]
            rus = RandomUnderSampler()
            X_test, y_test = rus.fit_resample(X_test, y_test)

            # train xgb model 
            model = GestureRecognitionModel("xgboost")
            model.train(X_train,y_train)

        else:
            df_sub = df_test[df_test.subject == subject]
            X_test, y_test = df_sub[features_to_keep], [gesture_to_int[gesture] for gesture in df_sub.gesture]
            rus = RandomUnderSampler()
            X_test, y_test = rus.fit_resample(X_test, y_test)

        # evaluate
        score = model.evaluate(X_test, y_test)
        
        c.setFont("Helvetica", 20)
        c.drawString(1 * inch, 10 * inch, f"{subject}: {score}")

        # confusion matrix
        preds = model.predict(X_test)
        cf_matrix = confusion_matrix(y_test, preds)
        fig = plt.figure(figsize = (10, 8))
        sns.heatmap(cf_matrix, annot=True, fmt=".0f")
        plt.yticks(np.arange(0.5,len(gestures_to_use)-1, 1), [int_to_gesture[i] for i in range(len(int_to_gesture))], rotation=45)
        plt.xticks(np.arange(0.5,len(gestures_to_use)-1, 1), [int_to_gesture[i] for i in range(len(int_to_gesture))], rotation=45)
        plt.ylabel("true")
        plt.xlabel("predicted")
        plt.tight_layout()
        
        # Save the plot as a temporary image file
        plt.savefig("tmp.png")
        plt.close(fig)

        # Add the image to the PDF
        img = ImageReader("tmp.png") 
        c.drawImage(img, 0.05*inch, y - 1 * inch, width=9 * inch, height=7 * inch)
        os.remove("tmp.png")

        c.showPage()
        y = 2 * inch
    
    # save result PDF
    c.save()

    

if __name__ == "__main__":
    main()