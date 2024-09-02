# -*- coding: utf-8 -*-
"""speech_emotion_recognition.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wGMkOEho2IonOmqYY250hsPPUgR2eFuZ
"""

from google.colab import files
uploaded = files.upload()

import zipfile
import os

with zipfile.ZipFile("/content/archive (1).zip", 'r') as zip_ref:
    zip_ref.extractall("/content/dataset")



import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

"""path to dataset"""

dataset_path = '/content/dataset'

# Load the dataset
paths = []
labels = []
for dirname, _, filenames in os.walk(dataset_path):
    for filename in filenames:
        if filename.endswith('.wav'):
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[-1].split('.')[0]
            labels.append(label.lower())

print('Dataset is loaded')
print(f"Total samples: {len(paths)}")
print(paths[:5])
print(labels[:5])

"""create dataframe"""

df = pd.DataFrame({'speech': paths, 'label': labels})
print(df.head())

# Label distribution
print("Label distribution:")
print(df['label'].value_counts())

"""Plotting label counts"""

sns.countplot(data=df, x='label')
plt.title("Label Distribution")
plt.show()

"""plot waveforms and spectrograms"""

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

"""Example of plotting for an emotion"""

emotion = 'fear'
emotion_data = df['speech'][df['label'] == emotion]

"""Check if there are any entries for the specified emotion"""

if len(emotion_data) > 0:
    path = np.array(emotion_data)[0]
    data, sampling_rate = librosa.load(path)
    waveplot(data, sampling_rate, emotion)
    spectogram(data, sampling_rate, emotion)
    display(Audio(path))
else:
    print(f"No data found for the emotion: {emotion}")

"""Feature extraction using MFCCs"""

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X = np.array([x for x in X_mfcc])
print(X.shape)

""" One-hot encode the labels"""

enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
y = y.toarray()
print(y.shape)

"""Split data into training and validation sets"""

X = np.expand_dims(X, -1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

"""Model definition"""

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # Adjust output layer based on the number of unique labels
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

"""Train the model"""

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=64)

""" Plot accuracy and loss"""

epochs = list(range(30))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""Predictions and evaluation"""

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)

"""Confusion matrix and classification report"""

conf_matrix = confusion_matrix(y_val_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

target_names = df['label'].unique()  # Dynamically get the labels for the target names
print("Classification Report:")
print(classification_report(y_val_classes, y_pred_classes, target_names=target_names))

""" Plot confusion matrix heatmap"""

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
