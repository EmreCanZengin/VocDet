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

def train_val_test_split(speech_speaker_df: pd.DataFrame, val_test_size: list[float] = [0.1, 0.1],random_state:int= 42, save_data = False ):
    X = speech_speaker_df["Speech"]
    y = speech_speaker_df["Speaker"]

    np.random.seed(random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= val_test_size[0], stratify= y, random_state= random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= val_test_size[1], stratify= y_train, random_state= random_state)
    if save_data:
        X_train.to_csv("X_train.txt")
        y_train.to_csv("y_train.txt")
        X_val.to_csv("X_val.txt")
        y_val.to_csv("y_val.txt")
        X_test.to_csv("X_test.txt")
        y_test.to_csv("y_test.txt")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_val_test_split_openset(speech_speaker_df: pd.DataFrame, val_test_size:list[float] = [0.1, 0.1], n_unclass_in_val_test:list[int] = [1, 2], random_state:int= 42, cc_val = False, save_data= False):

    classes = speech_speaker_df["Speaker"].unique()
    n_classes = len(classes)

    if sum(n_unclass_in_val_test) * 2 > n_classes:
        raise Exception("The number of classes that is wanted to used as unknown classes in validation and test class should be less than half of the total number of classes")

    np.random.seed(42)
    choosen_classes = np.random.choice(classes, size = sum(n_unclass_in_val_test))
    val_unknown_classes = choosen_classes[:n_unclass_in_val_test[0]]    
    test_unknown_classes = choosen_classes[n_unclass_in_val_test[0]:]

    val_mask = speech_speaker_df["Speaker"].isin(val_unknown_classes)
    test_mask = speech_speaker_df["Speaker"].isin(test_unknown_classes)

    val_unknown_df = speech_speaker_df[val_mask].copy()
    test_unknown_df = speech_speaker_df[test_mask].copy()

    if not cc_val:
        val_unknown_df["Speaker"] = -1

    test_unknown_df["Speaker"] = -1

    rest_mask= ~(val_mask | test_mask)
    rest_df = speech_speaker_df[rest_mask]

    X_train_known, X_test_known, y_train_known, y_test_known = train_test_split(rest_df["Speech"], rest_df["Speaker"], test_size= val_test_size[1], random_state= random_state,stratify= rest_df["Speaker"],shuffle= True)

    X_train_known, X_val_known, y_train_known, y_val_known = train_test_split(X_train_known, y_train_known, test_size= val_test_size[0],stratify=y_train_known, random_state= random_state, shuffle= True)


    X_val = pd.concat([X_val_known, val_unknown_df["Speech"]], axis= 0)
    y_val = pd.concat([y_val_known, val_unknown_df["Speaker"]], axis= 0)
    val_idx = np.arange(len(y_val))
    np.random.shuffle(val_idx)
    X_val_final = X_val.iloc[val_idx]
    y_val_final = y_val.iloc[val_idx]

    X_test = pd.concat([X_test_known, test_unknown_df["Speech"]], axis= 0)
    y_test = pd.concat([y_test_known, test_unknown_df["Speaker"]], axis= 0)

    test_idx = np.arange(len(y_test))
    np.random.shuffle(test_idx)
    X_test_final = X_test.iloc[test_idx]
    y_test_final = y_test.iloc[test_idx]
    if save_data: 
        X_train_known.to_csv("X_train_known.txt")
        y_train_known.to_csv("y_train_known.txt")

        if cc_val:
            X_val_final.to_csv("X_val_final_cc_val.txt")
            y_val_final.to_csv("y_val_final_cc_val.txt")
        else:
            X_val_final.to_csv("X_val_final.txt")
            y_val_final.to_csv("y_val_final.txt")

        X_test_final.to_csv("X_test_final.txt")
        y_test_final.to_csv("y_test_final.txt")
    return  X_train_known, X_val_final, X_test_final, y_train_known, y_val_final, y_test_final
