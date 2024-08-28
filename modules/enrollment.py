from modules.train import * # assuming we will write a model in train in a future.
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
from modules.record import *
import shutil

class Speaker:
    count = 0
    def __init__(self, speaker_name: str):
        Speaker.count += 1
        self.name = speaker_name
        self.id = Speaker.count
        self.speeches = np.empty(shape= (0,)) 
        self.n_speech = len(self.speeches)
        self.enrollments = []


class Enrollment(object):
    def __init__(self, name:str= "Unknown"):
        self.name = name
        self.known_speakers = {} 
        self.previous_best_model = None
        self.model = None
        self.preprocessing = None
    
    def addSpeaker(self, speaker_name: str):
        if not self.known_speakers.get(speaker_name, 0):
            speaker = Speaker(speaker_name)
            speaker.enrollments.append(self)
            self.known_speakers[speaker.name] = speaker
            print("The Speaker is added.")

    def addSpeech(self, speaker_name, file_name:Path):
        self.known_speakers[speaker_name].speeches = np.append(self.known_speakers[speaker_name].speeches, [file_name.absolute()], axis= 0)
        self.known_speakers[speaker_name].n_speech = len(self.known_speakers[speaker_name].speeches)

    def findClaimedId(self,name: str):
        for speaker in self.known_speakers:
            if speaker.name == name:
                return speaker.id
        else:
            raise Exception("Sorry you entered a wrong name. ")

    def findSpeakerNameWithId(self, id:int):
        for speaker in self.known_speakers:
            if speaker.id == id:
                return speaker.name
        else:
            raise Exception("Sorry you entered a wrong id. ")

    def updateModel(self):
        print(f"Updating {self.name} enrollment...")
        X_data, y_data = self.toDataframe()
        self.model, self.preprocessing = train(X_data, y_data, ccv= True)

    def deleteSpeaker(self,):
        pass

    def deleteEnrollment(self):
        # in future, admin speaker verification might be required but right now 
        del self

    def deleteSpeechs(self,speaker: Speaker):
        speaker_path = speaker.speeches[-1].parent

        enumerated_list = list(enumerate(speaker.speeches.values()))
        for idx, speech in enumerated_list:
            print(f"{idx}. {speech.relative_to(speaker_path)}")
        
        start = int(input("Starting index to delete: "))
        end = int(input("Last index to delete(inclusive): "))

        if start < 0 or end > len(speaker.speeches):
            print("Invalid index. Please try again.")
        else: 
           np.delete(self.known_speakers[speaker.name].speeches, np.s_[start:(end+1)], axis= 0) 
        
    def toDataframe(self):
        X_data = np.empty(shape= (0, ))
        y_data = np.empty(shape=(0,), dtype= np.int32)
        for speaker in self.known_speakers.values():
            X_data = np.concatenate([X_data, speaker.speeches], axis= 0)
            y_data = np.concatenate([y_data, np.full(shape= len(speaker.speeches), fill_value= speaker.id, dtype= np.int32) ], axis= 0)
        return X_data, y_data


    def cleanEnrollment(self):
        # admin verification will be required.
        self.known_speakers = {}
        self.model = None
