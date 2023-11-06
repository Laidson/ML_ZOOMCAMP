# import sys
# # Add your desired directory to the Python path
# sys.path.append('./src')
import pandas as pd
from sklearn.model_selection import GridSearchCV

from stage_02_train_model import ModelTraning


class ModelTuning:

    def __init__(self) -> None:
        pass
    

    def grid_search(self, pipeline, X_train, y_train):

        # Define a hyperparameter grid for grid search
        param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 0.5, 1, 10],  # Regularization parameter
        'classifier__max_iter': [100, 500],  # Maximum number of iteration
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver':['lbfgs']
        }

        # Create a GridSearchCV object
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc')
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters and best estimator
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_

        cv_results_df = pd.DataFrame(grid_search.cv_results_).round(3)

        return best_estimator, best_params


if __name__ == '__main__':
    mtu = ModelTuning()   

    df = ModelTraning().load_data_to_model()
    data_dict = ModelTraning().split_data(df)
    X = data_dict['X']
    y = data_dict['y']

    X_train = data_dict['X_train']
    X_test = data_dict['X_test']

    y_train  = data_dict['y_train']
    y_test = data_dict['y_test']
    pipeline = ModelTraning().model_pipeline(df=df)
    # grid search

    best_estimator, best_params = mtu.grid_search(pipeline=pipeline, X_train=X_train, y_train=y_train)
    # Make predictions using the best estimator
    y_pred = best_estimator.predict(X_test)

    ModelTraning().model_evaluation(y_pred=y_pred, y_test=y_test, path='artifacts/tuning_model', model_name='tuned_model')
    ModelTraning().save_model_informations(model=best_estimator, path='artifacts/tuning_model',model_name='tuned_model')

    print('** STAGE 03 - end **')


    