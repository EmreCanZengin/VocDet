from sklearn.metrics import recall_score, precision_score, accuracy_score, jaccard_score
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np


class ThresholdModel(ClassifierMixin, BaseEstimator):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def fit(self, X_train, y_train, onlyTrue= True, quant_ratio= 0.05):
        self.y_train_ = y_train.copy()
        self.X_train_ = X_train.copy()

        def threshold(y_proba):
            self.y_train_proba= y_proba.copy()
            n_classes = self.y_train_proba.shape[1]
#             if onlyTrue:
#                 class_idx = np.argmax(y_proba, axis= 1)
#                 mask = self.model.classes_[class_idx] == y_train
#                 self.y_train_proba = y_proba[mask]

#             if len(self.y_train_proba) < n_classes:
#                 raise ValueError("This model requires that both the number of true classifications \
# and the number of samples be at least equal to the number of known classes.")
            
            # th_vec = np.ones(shape= (n_classes,))
            # for class_ in range(n_classes):
            #     rows = np.argmax(self.y_train_proba, axis= 1) == class_
            #     class_proba = self.y_train_proba[rows, class_]
            #     if len(class_proba) == 0:
            #     # if there is no prediction to the given known class.
            #     # model might be poor.
            #         print(f"Model didn't predict: {self.classes[class_]}")
            #         continue
            #     th_vec[class_] = np.quantile(class_proba, quant_ratio)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(self.X_train_)
                max_probs = np.max(probs, axis=1)
                threshold = np.percentile(max_probs, quant_ratio * 100)
                return np.full(shape= (n_classes,), fill_value=threshold)
            else:
                raise ValueError("Modelin `predict_proba` metodu yok.")

            # return th_vec

        self.model.fit(self.X_train_,self.y_train_)
        self.classes = np.unique(self.y_train_) # I changed here.
        y_train_proba = self.model.predict_proba(self.X_train_)
        self.thresholds = threshold(y_train_proba)

        return self

    def predict(self, X_new):
        y_new_proba =  self.model.predict_proba(X_new)
        y_pred = -np.ones(len(X_new))
        max_idx = np.argmax(y_new_proba, axis= 1)
        y_max_proba = np.max(y_new_proba, axis = 1)

        probable_classes = self.classes[max_idx]
        mask = y_max_proba >= self.thresholds[max_idx]
        y_pred[mask] = probable_classes[mask]
        return y_pred

    def report(self, y_true, y_pred, when= "Training", average= "macro"):

        mask_unknown = y_true == -1
        y_true_unknown = y_true[mask_unknown]
        y_true_known = y_true[~mask_unknown]

        y_pred_unknown = y_pred[mask_unknown]
        y_pred_known = y_pred[~mask_unknown]

        n_pred_unknown = np.sum(y_pred == -1)
        n_true_unknown = np.sum(mask_unknown)


        accuracy = accuracy_score(y_pred=y_pred_known, y_true=y_true_known)
        recall = recall_score(y_pred=y_pred_known, y_true=y_true_known, average=average, zero_division= np.nan)
        precision = precision_score(y_pred=y_pred_known, y_true=y_true_known, average=average, zero_division= np.nan)
        jaccard = jaccard_score(y_true= y_true_known, y_pred= y_pred_known, average= average, zero_division= 1)

        recall_unknown = 0.0 
        precision_unknown = 0.0
        accuracy_unknown = 0.0
        jaccard_unknown = 0.0
        rat_pred_true_unknown = 0.0

        if len(y_pred_unknown)>0:
            recall_unknown = recall_score(y_pred=y_pred_unknown, y_true=y_true_unknown, average=average, zero_division= np.nan)
            precision_unknown = precision_score(y_pred=y_pred_unknown, y_true=y_true_unknown, average=average,zero_division=np.nan)
            jaccard_unknown = jaccard_score(y_true= y_true_unknown, y_pred= y_pred_unknown, average= average, zero_division= 1.0)
            accuracy_unknown= accuracy_score(y_pred=y_pred_unknown, y_true=y_true_unknown)
            rat_pred_true_unknown = n_pred_unknown / n_true_unknown 


        else:
            print("The model could not realize any unknown data. There might be no unknown class.")

        model_name = repr(self.model)

        report = f"""{model_name} Classifier for {when} Data in Open Set:
For Known Data Points:
            Accuracy: {accuracy}
            Recall: {recall}
            Precision: {precision}
            Intersection: {jaccard}
For Unknown Data Points:
            Accuracy: {accuracy_unknown}
            Recall: {recall_unknown}
            Precision: {precision_unknown}
            Intersection: {jaccard_unknown}
            Custom Unknown Score: {rat_pred_true_unknown}\n"""
        
        return {"accuracy_score": accuracy, "recall_score": recall, "precision_score": precision, "jaccard_score": jaccard,
                "accuracy_score_unknown": accuracy_unknown, "recall_score_unknown": recall_unknown, "precision_score_unknown": precision_unknown, "jaccard_score_unknown": jaccard_unknown,
                 "custom_unknown_score": rat_pred_true_unknown, "report": report}

    def set_thresholds(self, thresholds: np.ndarray)->None:
        try:
            if len(thresholds) == len(self.classes):
                self.thresholds = thresholds.copy()
            else:
                raise ValueError("The threshold array is not compatible with the known class size")
        except:
            raise Exception("The model has not been fitted yet. Please fit the model first.")
