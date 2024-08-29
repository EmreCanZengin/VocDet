from typing import Union, Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from modules.threshold import ThresholdModel
from sklearn.datasets import load_iris


def cross_class_validation(model, X_data: Union[np.ndarray, pd.DataFrame], y_data: Union[np.ndarray, pd.DataFrame],k_folds: int, random_state= 42)-> \
    Tuple[Dict[str, List[float]], np.ndarray, object]:

    def update_dictionary(src: Dict[str, object], dest: Dict[str, object]) -> None:
        for key in dest.keys():
            dest[key] = src[key]
    
    best_model = None

    if k_folds < 1:
        raise ValueError("Value must be positive integer.")

    try:
        best_model = None
        best_score = 0.0
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

        # should be calculated from k_class_folds
        n_known_classes = k_class_folds - 1;

        score_types = ["accuracy_score", "recall_score", "precision_score", "jaccard_score",
                       "accuracy_score_unknown", "recall_score_unknown", "precision_score_unknown", "jaccard_score_unknown"]
        scores_dict = {score_type: 0.0 for score_type in score_types}
      
        for k in range(k_class_folds):
            unknown_classes = classes[k]
            idx_unknown = y == unknown_classes

            # take the unknown class fold
            X_unknown = X[idx_unknown]
            y_unknown = y[idx_unknown]

            X_known = X[~idx_unknown]
            y_known = y[~idx_unknown]

            n_known_classes = len(np.unique(y_known))
            for j in range(k_folds):
                print(f"Model index: {k}-{j}")
                X_train, X_val_known, y_train, y_val_known = train_test_split(X_known, y_known,stratify=y_known, random_state= j * 20, test_size= val_ratio)

                # no need for validation to shuffle but requires to transform (-1)
                X_val = np.concatenate([X_val_known, X_unknown], axis= 0)
                y_unknown[:] = -1
                y_val = np.concatenate([y_val_known, y_unknown], axis= 0)
                
                th_model = ThresholdModel(model)
                th_model = th_model.fit(X_train, y_train)

                y_pred = th_model.predict(X_val)
                src_dict = th_model.report(y_pred= y_pred, y_true= y_val)
                update_dictionary(src= src_dict, dest= scores_dict)
                scores = np.array(list(scores_dict.values()))

                if best_model is None:
                    best_model = th_model

                curr_norm = np.linalg.norm(scores)
                prod = np.prod(scores)
                if curr_norm > best_score and prod > 0:
                    best_score = curr_norm
                    best_model = th_model

        return best_model

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
