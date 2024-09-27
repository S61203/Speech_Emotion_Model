**Speech Emotion Recognition**

This project is a Speech Emotion Recognition (SER) model that classifies speech audio files into different emotions using an LSTM-based neural network. The dataset used in this project is the Toronto Emotional Speech Set (TESS).

**Table of Contents**

* Dataset
* Usage
* Model Architecture
* Results

**Dataset**

The dataset used is the Toronto Emotional Speech Set (TESS), which consists of 200 target words spoken in a neutral North American accent by two actresses (aged 26 and 64) and recorded in seven different emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral.
The dataset can be found on Kaggle.

**Usage**

Upload the dataset to your Google Colab environment.
Run the speech_emotion_recognition.ipynb notebook to execute the model pipeline:
Loading and exploring the dataset.
Visualizing audio waveforms and spectrograms.
Extracting MFCC features from the audio data.
Building and training an LSTM-based model.
Evaluating the model's performance using a confusion matrix and classification report.

**Model Architecture**

The model is built using the Keras library and consists of the following layers:

* LSTM Layer: 256 units, followed by a dropout layer.
* Dense Layer: 128 units with ReLU activation, followed by a dropout layer.
* Dense Layer: 64 units with ReLU activation, followed by a dropout layer.
* Output Layer: 7 units with Softmax activation (one for each emotion class).
* The model is trained using categorical cross-entropy loss and the Adam optimizer.

**Results**
The model was trained for 30 epochs with a batch size of 64. 

The following results were obtained:

* Training Accuracy: 0.9949
* Validation Accuracy: 0.9821

**Confusion Matrix:** 

A heatmap representation of the confusion matrix.

**Classification Report:** 

Precision, recall, and F1-score for each emotion class.

**References**
* Dataset: Toronto Emotional Speech Set (TESS) - Link to Dataset
* Libraries: Keras, Librosa, Seaborn, Matplotlib, Scikit-learn
