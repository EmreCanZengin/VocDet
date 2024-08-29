from modules.Data_Load import *
from modules.Data_Processing import *
from modules.threshold import *
from modules.Cross_Class_Validation import * 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


PROGRAM_VERSION = "0.0.1"
CURR_PATH = Path.cwd().absolute()
ZIP_PATH = CURR_PATH / "archive.zip"
DATA_PATH = CURR_PATH / "data"

def Report(best_model):
    # svc document said predict_proba and predict can be inconsistent
    y_train_pred = best_model.predict(X_train_transformed)
    train_report = best_model.report(y_train, y_train_pred, when="Training")["report"]
    y_test_pred = best_model.predict(X_test_transformed)
    test_report = best_model.report(y_test, y_test_pred, when="Test")["report"]
    print(train_report, test_report, sep="\n")

if __name__ == "__main__":
    print("Welcome to Project VocDet")
    print("VocDet version: ", PROGRAM_VERSION)

    extractZipToDataFolder(ZIP_PATH, DATA_PATH)
    speech_speaker_df = dataLoader(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_openset(speech_speaker_df, cc_val=True)

    X_train = pd.concat([X_train, X_val], axis= 0)
    y_train = pd.concat([y_train, y_val])

    pre_processing_pipe = Pipeline(steps= [
        ("fe_librosa", FeatureExtractionWithLibrosa()),
        ("pca", PCAUncompatibleShapes(row_mfcc= 13)), 
        ("flatten", FunctionTransformer(flattenLastTwoDim)),
        # ("scaler", StandardScaler())
        ("scaler", MinMaxScaler(feature_range=(0, 1))),
    ])

    print("Preprocessing data.\n[WARNING] This will take some time.")
    X_train_transformed = pre_processing_pipe.fit_transform(X_train)
    X_test_transformed = pre_processing_pipe.fit_transform(X_test)
    print("Preprocessing the data: SUCCESS")

    

    svc = SVC(C= 5,kernel= "linear" , probability= True)
    rf = RandomForestClassifier(n_estimators=10, n_jobs= -1, min_samples_split=10)
    knn = KNeighborsClassifier(n_neighbors= 10, n_jobs= -1)

    print("SVC")
    svc_scores_dict, svc_thresholds, svc_best_model = cross_class_validation(svc, X_train_transformed, y_train, k_folds= 3)

    print("RF")
    rf_scores_dict, rf_thresholds, rf_best_model = cross_class_validation(rf, X_train_transformed, y_train, k_folds= 3)

    print("KNN")
    knn_scores_dict, knn_thresholds, knn_best_model = cross_class_validation(knn, X_train_transformed, y_train, k_folds= 3)
    
    Report(svc_best_model)
    Report(rf_best_model)
    Report(knn_best_model)


