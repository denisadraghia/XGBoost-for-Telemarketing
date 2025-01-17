import xgboost as xgb
from sklearn.model_selection import GridSearchCV
def fit_XGBoost(X_train, y_train, params, scoring_metric='f1', scale_pos_weight=1):
    """
    Fits an XGBoost model using GridSearchCV with the provided parameters and scoring metric.

    Parameters:
    - X_train: Features for training
    - y_train: Target variable
    - params: Dictionary of hyperparameters to tune
    - scoring_metric: Metric for scoring (default is 'f1')
    - scale_pos_weight: Weight for the positive class to handle imbalance (default is 1)

    Returns:
    - best_model: The best XGBoost model found by GridSearchCV
    - best_params: Best parameters found during GridSearchCV
    """
    
    # Initialize the XGBoost classifier with the provided scale_pos_weight
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic',  
                                 scale_pos_weight=scale_pos_weight)
    
    # Set up GridSearchCV with the provided parameters and scoring metric
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=params, scoring=scoring_metric, 
                               cv=5, verbose=0, n_jobs=-1)
    
    # Fit the model on the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    
    
    return best_model