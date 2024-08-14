import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

dataset_dir = r"C:\Users\melis\OneDrive\Masaüstü\VocDetCode\50_speakers_audio_data"

def extract_mfcc_features(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

all_mfcc_features = []

for subdir, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(subdir, file)
            mfcc_features = extract_mfcc_features(file_path)
            all_mfcc_features.append(mfcc_features)

all_mfcc_features = np.array(all_mfcc_features)


plt.figure(figsize=(12, 6))
sns.heatmap(all_mfcc_features, cmap='viridis', cbar=True)
plt.title('MFCC')
plt.xlabel('MFCC')
plt.ylabel('Ses Dosyaları')
plt.show()

print(all_mfcc_features.shape)

all_mfcc_features_normalized = minmax_scale(all_mfcc_features, feature_range=(0, 1))

plt.figure(figsize=(12, 6))
sns.heatmap(all_mfcc_features_normalized, cmap='viridis', cbar=True)
plt.title('MFCC Min-Max Normalizasyonu')
plt.xlabel('MFCC')
plt.ylabel('Ses Dosyaları')
plt.show()

n_components = 2
pca = PCA(n_components=n_components)
all_mfcc_features_pca = pca.fit_transform(all_mfcc_features_normalized)

plt.figure(figsize=(12, 6))
plt.scatter(all_mfcc_features_pca[:, 0], all_mfcc_features_pca[:, 1], c='blue', edgecolor='k', alpha=0.5)
plt.title(f'PCA Dönüşümü (Top {n_components} Components)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()

labels = []

for subdir, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith('.wav'):
            label = int(os.path.basename(subdir)[-4:])
            labels.append(label)

X_train, X_test, y_train, y_test = train_test_split(all_mfcc_features_pca, labels, test_size=0.2, random_state=42)

linear_svc_model = LinearSVC(C=1.0, max_iter=1000, random_state=42)
linear_svc_model.fit(X_train, y_train)
y_pred = linear_svc_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.barplot(x=["Accuracy"], y=[accuracy * 100])
plt.ylim(0, 100)
plt.title("Model Accuracy")
plt.ylabel("Accuracy (%)")
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Karışıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()
