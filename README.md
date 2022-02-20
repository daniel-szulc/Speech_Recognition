<p align="center">
  <h1 align="center">Speech Recognition</h1></p>
<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat)](https://www.python.org/)
[![TensorFlow Version](https://img.shields.io/badge/Tensorflow-2.5-blue.svg?style=flat)](https://www.tensorflow.org/) 

</div>
The project aims to detect single-word commands. It allows you to recognize keywords continuously in real time and return a match.

## Get started

To start the speech recognition process, enter "ACTIVE_ON" on the console.

Using a trained model, you can directly use the <b>speechRecognition.py</b> script to recognize the voice samples continuously.
Recognized samples must contain these trained words:

for the English model:

- "left"
- "right"
- "jump"
- "stop"
- "shoot"
- "attack"

for the Polish model:

- "lewo"
- "prawo"
- "skocz"
- "stop"
- "strzał"
- "atak"

## Configuration

You can use the following commands to operate the speech recognition interface:

| Command  | Operation |
| ------------- | ------------- |
| <code>ACTIVE_ON</code> |  Activation of the speech recognition system |
| <code>ACTIVE_OFF</code> | Pause of the speech recognition system |
| <code>LANGUAGE_EN</code> | Changing the current model to the English language model |
| <code>LANGUAGE_PL</code> | Changing the current model to the Polish language model |
| <code>GET_STATE</code> | Returns the current system status (None/LOADING/READY) |
| <code>SET_SILENCE</code> | Automatically set the microphone sensitivity |
| <code>GET_SILENCE</code> | Returns the current set microphone sensitivity |
| <code>SET_dB:YOUR_VALUE</code> | Set the microphone sensitivity for the decibel level |
| <code>SET_RMS:YOUR_VALUE</code> | Set the microphone sensitivity for the root mean square level |

## About the dataset

Samples from the following sources were used to create speech classifying models:

- Google Speech Commands Dataset
https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.zip <br>
The Google Speech Command dataset created by the TensorFlow and AIY teams to provide an example of speech recognition using the TensorFlow API. The dataset contains thousands of different clips that are one second long.

- a set made by text-to-speech using Google Text-to-Speech API
https://cloud.google.com/text-to-speech

- a set made during self-conducted voice recordings. Several people of different ages participated in the recordings.

The prepared set of 2 437 audio files was divided into training, validation and test sets, using the ratio of 80:10:10, respectively.

## Result

Graphs from model learning processes are presented below:



| <img src="/img/result_en.png"/> | <img src="/img/result_pl.png" /> |
| ------------- | ------------- |
| <div align="center">English model</div> | <div align="center">Polish model</div> |




## Source

The code was based on Simple audio recognition: Recognizing keywords from TensorFlow.
https://www.tensorflow.org/tutorials/audio/simple_audio