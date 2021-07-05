<p align="center">
  <h1 align="center">Speech Recognition</h1></p>

[![MIT Licensed](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](https://opensource.org/licenses/MIT) 
[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat)](https://www.python.org/)
[![TensorFlow Version](https://img.shields.io/badge/Tensorflow-2.5-blue.svg?style=flat)](https://www.tensorflow.org/) 

The project aims to detect single-word commands. It allows you to recognize keywords continuously in real time and return a match.

## Get started

Using a trained model, you can directly use the speechRecognition.py script to record a single voice sample, or the realtimeSpeechRecognition.py script to recognize the voice samples continuously.
Recognized samples must contain these trained words:

- "down"
- "forward"
- "left"
- "right"
- "stop"

## About the dataset

Google Speech Commands Dataset

https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.zip

The Google Speech Command dataset created by the TensorFlow and AIY teams to provide an example of speech recognition using the TensorFlow API. The dataset contains thousands of different clips that are one second long.