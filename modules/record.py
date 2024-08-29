import numpy as np
import pandas as pd
import keyboard
import time
import sounddevice as sd
from scipy.io.wavfile import write


def record_audio(file_path, duration_sec=30, fs=44_100):
    print("Kayıt başlatmak için q tuşuna basın...")

    keyboard.wait('q')
    
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    print("Kayıt başlıyor...")
    # print("ask questionGenerator.question() to answer")
    # print("Please answer this question about a minute.")
    # print("Time Indicator .....")
    # print("Time is done. close recording")

    audio = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Kayıt tamamlandı.")
    write(file_path, fs, audio)

