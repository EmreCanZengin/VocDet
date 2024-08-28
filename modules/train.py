import numpy as np
from pathlib import Path
from modules.threshold import *
from modules.Cross_Class_Validation import *
from modules.Data_Load import *
from modules.Data_Processing import *
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier

def train(X_data: np.ndarray[Path], y_data:np.ndarray[int], ccv= False):

    # preprocessing
    # training for closed set

    # training with ccv
    # training without ccv
    # reporting the model
    pre_processing_pipe = Pipeline(steps= [
        ("fe_librosa", FeatureExtractionWithLibrosa()),
        ("pca", PCAUncompatibleShapes(row_mfcc= 13)), 
        ("flatten", FunctionTransformer(flattenLastTwoDim)),
        # ("scaler", StandardScaler())
        ("scaler", MinMaxScaler(feature_range=(0, 1))),
    ])
    n_classes = len(np.unique(y_data))
    df = pd.DataFrame({ "Speech": X_data, "Speaker": y_data.astype(np.int32) })
    print("Processing the Data.")
    X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size= 0.1, stratify= y_data, random_state= 42)
    X_train_transformed = pre_processing_pipe.fit_transform(X_train, y_train)
    X_test_transformed = pre_processing_pipe.fit_transform(X_test)

    model_sub = chooseClosedSetModel(X_train_transformed, X_test_transformed, y_train, y_test)
    model = ThresholdModel(model_sub)

    if ccv and n_classes > 10: 
        print("Starting to Cross class Validation")
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_openset(df, cc_val= ccv)
        X_train = np.concatenate([X_train, X_val], axis= 0)
        y_train = np.concatenate([y_train, y_val], axis= 0, dtype= np.int32)
        X_train_transformed = pre_processing_pipe.fit_transform(X_train, y_train)
        X_test_transformed = pre_processing_pipe.fit_transform(X_test)
        
        best_model = cross_class_validation(model_sub, X_train_transformed, y_train, k_folds=3)
        y_train_pred = best_model.predict(X_train_transformed)
        train_report = best_model.report(y_train, y_train_pred, when="Training")["report"]
        y_test_pred = best_model.predict(X_test_transformed)
        test_report = best_model.report(y_test, y_test_pred, when="Test")["report"]
        print(train_report, test_report, sep="\n")
        return best_model, pre_processing_pipe

    else:
        print("Think about here")

def chooseClosedSetModel(X_train, X_test, y_train, y_test):
    print("\nChoosing a model") # is there a leakage here
    # I don't know how i should choose grids here
    models = {
        'rf': RandomForestClassifier(),
        'svc': SVC(),
    }
    param_grids = {
        'rf': {'n_estimators': [10, 50, 100], 'min_samples_split': [5, 10], 'max_depth': [50]},
        'svc': {'C': [0.1, 1, 5, 10, 25, 50], "probability": [True], 'kernel': ["linear", "rbf"]},
    }

    results = {}
    norm_accuracy_closedset = make_scorer(closedsetScoring, greater_is_better= True)
    for model_name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring=norm_accuracy_closedset)
        grid_search.fit(X_train, y_train)

        results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
    ## choose the best model in validation set.
    best_model = None
    best_score = 0.0
    for model_name, result in results.items():
        model = models[model_name]
        model.set_params(**result['best_params'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        new_score = accuracy_score(y_test, y_pred)
        if  new_score > best_score and new_score > 0.0:
            best_model = model
            best_score = new_score
    print(f"Best Model: {repr(best_model)}") 
    return best_model

def closedsetScoring(y_true, y_pred, average= "macro"):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average= average, zero_division= np.nan)
    pre = precision_score(y_true, y_pred, average= average, zero_division= np.nan)

    return np.linalg.norm(np.array([acc, rec, pre]))





    
