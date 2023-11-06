import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix



class ModelTraning:

    def __init__(self) -> None:
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def load_data_to_model(self, parquet_file_path='data\data_to_model\data_model.parquet'):

        df = pd.read_parquet(parquet_file_path, engine='fastparquet')  
        return df
    
    def split_data(self, df):

        X = df[['dayofweek', 'time', 'length','international']]
        y = df['delay']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        return {'X':X, 'y':y, 'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}


    def model_pipeline(self, df):

        cat_cols = df.select_dtypes(include=['object']).columns
        num_cols = df.select_dtypes(include=['int64']).columns
        bin_cols = df.select_dtypes(include=['int8']).columns.drop('delay')

        # Create a ColumnTransformer to apply StandardScaler to numerical features and OneHotEncoder to categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(drop='first'), cat_cols),
            ],
            remainder='passthrough'  # Include 'bin' as a numerical feature without scaling or encoding
        )

        # Create a scikit-learn pipeline
        pipeline = Pipeline([
            # Apply the preprocessor (ColumnTransformer)
            ('preprocessor', preprocessor),
            # Add any additional steps to your pipeline, such as a classifier or regressor
            ('classifier', LogisticRegression(C=0.5,max_iter=100,penalty='l2',solver="lbfgs"),)
        ])

        return pipeline

    def train_model(self, pipeline, X_train, y_train):
        pipeline.fit(X_train, y_train)
        return pipeline

    def model_evaluation(self, y_test, y_pred, path, model_name):

        # Calculate and print various evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        # Create and display a confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        result_dict = {

            # Print the evaluation metrics
            "Accuracy": accuracy.round(2),
            "Precision": precision.round(2),
            "Recall": recall.round(2),
            "F1-Score": f1.round(2),
            "ROC-AUC Score": roc_auc.round(2),
            "Confusion Matrix":np.array(conf_matrix, dtype=int).tolist()
        }

        model_eva_path = f'{path}/evaluation_{model_name}.json'       
        with open(model_eva_path, 'w') as f:
            json.dump(result_dict, f, indent=4)

        return result_dict
    
    def save_model_informations(self, model, path, model_name):

        # Save the trained model
        model_filename = f'{path}/trained_{model_name}.pkl'
        joblib.dump(model.named_steps['classifier'], model_filename)

        # Save the preprocessing pipeline (including feature engineering steps)
        preprocessor_filename = f'{path}/preprocessing_pipeline_{model_name}.pkl'
        joblib.dump(model.named_steps['preprocessor'], preprocessor_filename)
        
        # Save the trained model params
        params_filename = f'{path}/params_{model_name}.json'
        model_parms = model.named_steps['classifier'].get_params()
        with open(params_filename, 'w') as f:
            json.dump(model_parms, f, indent=4)

        return




    
if __name__ == '__main__':

    mt = ModelTraning()

    df = mt.load_data_to_model()

    data_dict = mt.split_data(df)
    X = data_dict['X']
    y = data_dict['y']

    X_train = data_dict['X_train']
    X_test = data_dict['X_test']

    y_train  = data_dict['y_train']
    y_test = data_dict['y_test']

    pipeline = mt.model_pipeline(df)

    model = mt.train_model(pipeline,X_train,y_train,)
    y_pred = model.predict(X_test)
  
    
    #Saving 
    mt.model_evaluation(y_pred, y_test,path='artifacts/base_line_model', model_name='base_model')
    mt.save_model_informations(model, path='artifacts/base_line_model', model_name='base_model')
 
    print('** STAGE 02 - end **')
    



