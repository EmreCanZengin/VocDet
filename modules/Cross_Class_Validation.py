from typing import Union, Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from modules.threshold import ThresholdModel
from sklearn.datasets import load_iris


def cross_class_validation(model, X_data: Union[np.ndarray, pd.DataFrame], y_data: Union[np.ndarray, pd.DataFrame],k_folds: int, random_state= 42)-> \
    Tuple[Dict[str, List[float]], np.ndarray, object]:
    """
    ########################<br>
    # TODO Write a description here<br>
    ########################<br>
    problem: i am not sure how to split classes for validation. is just one enough or not

    params: 
    model: a model that has these methods or attributes: `predict_proba()`, `fit()`
    X_data: training data which might not be splited to training and validation data
    y_data: training data labels which might not be splited to training and validation data
    k_folds: the number of folds in cross validation part
    random_state:

    return:
    scores_dict: the dictionary that contains accuracy, recall, precision, and jaccard scores of the given model
    thresholds: overall threshold values
    best_model: best threshold model
    """
    def update_dictionary(src: Dict[str, object], dest: Dict[str, object], i: int, j: int) -> None:
        for key in dest.keys():
            dest[key][i, j] = src[key]
    
    best_model = None

    if k_folds < 1:
        raise ValueError("Value must be positive integer.")

    try:
        best_model = None
        # shuffle the dataset
        np.random.seed(random_state)
        idx = np.arange(X_data.shape[0])
        np.random.shuffle(idx)

        if isinstance(X_data, (pd.DataFrame, pd.Series)):
            X = X_data.iloc[idx].to_numpy()
        else:
            X = X_data[idx]  # Assuming it's already a NumPy array

        if isinstance(y_data, (pd.DataFrame, pd.Series)):
            y = y_data.iloc[idx].to_numpy().ravel()  # ravel() in case y is a DataFrame and needs flattening
        else:
            y = y_data[idx]

        # when we k_class_folds this might change
        classes = np.unique(y)
        k_class_folds = len(classes) 
        val_ratio = 1/ k_folds
        n_instances = len(X_data)

        # should be calculated from k_class_folds
        n_known_classes = k_class_folds - 1;
        dummy = np.zeros(shape= (k_class_folds, k_folds))
        thresholds = np.ones(shape= (k_class_folds, k_folds, n_known_classes))

        score_types = ["accuracy_score", "recall_score", "precision_score", "jaccard_score",
                       "accuracy_score_unknown", "recall_score_unknown", "precision_score_unknown", "jaccard_score_unknown"]
        scores_dict = {score_type: dummy.copy() for score_type in score_types}
      
        for k in range(k_class_folds):
            unknown_classes = classes[k]
            idx_unknown = y == unknown_classes

            # take the unknown class fold
            X_unknown = X[idx_unknown]
            y_unknown = y[idx_unknown]

            X_known = X[~idx_unknown]
            y_known = y[~idx_unknown]

            idx_instances = np.arange(len(X_known))

            n_known_classes = len(np.unique(y_known))
            upper_thresholds = np.ones(shape= (k_folds, n_known_classes))

            for j in range(k_folds):
                print(f"Model index: {k}-{j}")
                X_train, X_val_known, y_train, y_val_known = train_test_split(X_known, y_known,stratify=y_known, random_state= j * 20, test_size= val_ratio)

                # no need for validation to shuffle but requires to transform (-1)
                X_val = np.concatenate([X_val_known, X_unknown], axis= 0)
                y_unknown[:] = -1
                y_val = np.concatenate([y_val_known, y_unknown], axis= 0)
                
                th_model = ThresholdModel(model)
                th_model = th_model.fit(X_train, y_train)
                local_threshold = th_model.thresholds
                upper_thresholds[j] = local_threshold

                y_pred = th_model.predict(X_data)
                src_dict = th_model.report(y_pred= y_pred, y_true= y_data)
                update_dictionary(src= src_dict, dest= scores_dict, i= k, j= j)
                if best_model is None:
                    best_model = th_model
                    best_model_norm = np.linalg.norm(np.array(list(scores_dict.values())))
                else:
                    curr_norm = np.linalg.norm(np.array(list(scores_dict.values())))
                    if curr_norm > best_model_norm:
                        best_model = th_model
                    

            thresholds[k] = upper_thresholds
        return scores_dict, thresholds, best_model

    except Exception as e:
        print(f"An error is occured: {str(e)} in cross class validation function")
        raise 


if __name__ == "__main__":
    # Make sure to import modules correctly specifically in this folder.
    svc = SVC(C= 50, probability= True)
    data = load_iris()
    X_train = data.data
    y = data.target
    print(cross_class_validation(svc, X_train, y, 2))
