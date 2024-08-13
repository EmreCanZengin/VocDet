from modules.Data_Load import *

PROGRAM_VERSION = "0.0.1"
CURR_PATH = Path.cwd().absolute()
ZIP_PATH = CURR_PATH / "archive.zip"
DATA_PATH = CURR_PATH / "data"


if __name__ == "__main__":
    print("Welcome to Project VocDet")
    print("VocDet version: ", PROGRAM_VERSION)

    extractZipToDataFolder(ZIP_PATH, DATA_PATH)
    speech_speaker_df = dataLoader(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(speech_speaker_df)

    print(f"Shapes of Datasets\n\
X_train, y_train : {X_train.shape}, {y_train.shape},\n\
X_val, y_val: {X_val.shape}, {y_val.shape},\n\
X_test, y_test : {X_test.shape}, {y_test.shape}\n")

    print(speech_speaker_df.head(5))

    

    

