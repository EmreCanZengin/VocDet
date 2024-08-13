from zipfile import ZipFile
from pathlib import Path
import pandas as pd 
import numpy as np
from typing import Union
import shutil
from sklearn.model_selection import train_test_split


def extractZipToDataFolder(zip_path: Union[str, Path], to_extract: Union[str, Path]= "../data") -> None:
    print("\nUnzipping the data")
    answer = 0
    if isinstance(zip_path, str):
        zip_path = Path(zip_path).absolute()
    
    if isinstance(to_extract, str):
        to_extract = Path(to_extract).absolute()

    if to_extract.exists():
        answer = int(input("\nThe path you wanted to extract the archive exists,\n\
If you want to use the data currently available, press 1\n\
Otherwise The program will delete and unzip again: "))
        if answer != 1:
            print("\nRemoving the current data")
            shutil.rmtree(to_extract)
            print("Removing the data: SUCCESS")
            print("Continuing to unzip")

    if answer != 1:
        with ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(to_extract)

        ls_data_path = [f for f in to_extract.iterdir()]
        if len(ls_data_path) > 1:
            raise Exception("This function can't extract zip files with multiple folders in root")
        
        inner_folder = ls_data_path[0]
        for file_path in inner_folder.iterdir():
            file_name = file_path.relative_to(inner_folder)
            dest_path = to_extract / file_name

            if file_path.is_dir():
                shutil.move(file_path, dest_path)
        
        inner_folder.rmdir()
    print("Unzipping the data: SUCCESS\n")

def dataLoader(data_path: Union[Path, str] = "../data") -> pd.DataFrame:
    if isinstance(data_path, str):
        data_path = Path(data_path)
    speech_paths_labels = np.array([[speech.as_posix(), str(speaker.relative_to(data_path))] for speaker in data_path.iterdir() for speech in speaker.iterdir()])
    df = pd.DataFrame(speech_paths_labels, columns= ["Speech", "Speaker"])

    def mappingFunc(x:str):
        num = int(x[-4:])
        return num

    df["Speaker"] = df["Speaker"].map(mappingFunc)

    return df

def train_val_test_split(speech_speaker_df: Union[pd.DataFrame, np.ndarray], sizes: list[float] = [0.1, 0.1],random_state= 42 ):
    X = speech_speaker_df["Speech"]
    y = speech_speaker_df["Speaker"]

    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= sizes[0], stratify= y, random_state= random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= sizes[1], stratify= y_train, random_state= random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test
