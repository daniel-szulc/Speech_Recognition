from pyaudio import paInt16
from pyaudio import PyAudio
from soundfile import write as sf_write
from librosa import core as l_core
from os import path
from librosa import effects as l_effects
from librosa import get_duration as l_get_duration
from statistics import mean
from tensorflow import *
import gc
import tensorflow as tf
from tensorflow.keras import models
from pydub import AudioSegment, effects
import numpy as np
import sys
import multitasking
import tempfile
import audioop
import time

MODEL_EN = 'enModel.h5'
MODEL_PL = 'plModel.h5'
COMMANDS_EN = ['attack', 'jump', 'left', 'right', 'shoot', 'stop', 'unknown']
COMMANDS_PL = ['atak', 'lewo', 'prawo', 'skocz', 'stop', 'strzal']
AUTOTUNE = tf.data.AUTOTUNE
CHUNK = 1024
FORMAT = paInt16
CHANNELS = 1
RATE = 16000
RECORD_TIME = 2
SILENCE_dB = 35.44068044350276
SILENCE_RMS = 3500
IS_ACTIVE = False
LANGUAGE = "EN"  # "PL"
STATE = "None"

tempdirpath = tempfile.mkdtemp()

def get_spectrogram(audio):

    zero_padding = tf.zeros(16000 - tf.shape(audio), dtype=tf.float32)

    waveform = tf.cast(audio, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)

    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram, num_parallel_calls=AUTOTUNE)


    return output_ds


def get_waveform(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform


def edit_audio(audio):
    audio = audio.set_channels(1)  # to mono audio
    normalized_sound = effects.normalize(audio)  # normalize volume of audio
    y = np.array(normalized_sound.get_array_of_samples()).astype(np.float32)
    test = l_core.power_to_db(y)
    print(test)
    print(test.mean())
    y = y / (1 << 8 * 2 - 1)  # convert to librosa
    new_audio, index = l_effects.trim(y, top_db=SILENCE_dB)  # cut silence
    duration = l_get_duration(new_audio, RATE)
    if duration > 0.99:
        new_audio = l_effects.time_stretch(new_audio, duration + 0.01)
    return new_audio


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def edit_audio(audio):
    audio = audio.set_channels(1)  # to mono audio
    normalized_sound = effects.normalize(audio)  # normalize volume of audio
    y = np.array(normalized_sound.get_array_of_samples()).astype(np.float32)
    y = y / (1 << 8 * 2 - 1)  # convert to librosa
    new_audio, index = l_effects.trim(y, top_db=SILENCE_dB)  # cut silence
    duration = l_get_duration(new_audio, RATE)
    if duration > 0.99:
        new_audio = l_effects.time_stretch(new_audio, duration + 0.01)
    return new_audio


def get_silence_db():
    global IS_ACTIVE
    active_before = IS_ACTIVE
    IS_ACTIVE = False
    peaks = []
    rmss = []
    print('PLEASE BE QUIET...')
    for _ in range(0, int(RATE / CHUNK * 1)):
        data = stream.read(CHUNK)
        rms = audioop.rms(data, 2)
        if(rms>10):
            rmss.append(rms)
        data1 = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        peak = np.average(np.abs(data1)) * 2
        peaks.append(peak)
    print('DONE.')
    rms=mean(rmss)
    global SILENCE_dB
    global SILENCE_RMS
    rms = rms*1.1+2000
    rms = rms*1.1
    SILENCE_RMS = rms
    SILENCE_dB = 10 * np.log10(rms)
    print("SILENCE_dB:", SILENCE_dB)
    print("SILENCE_RMS:", SILENCE_RMS)
    IS_ACTIVE = active_before

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def asrRunner():

    frames = []
    for _ in range(0, int(RATE / CHUNK * 1)):
        data = stream.read(CHUNK)
        frames.append(data)
    sound = AudioSegment(
        data=b''.join(frames),
        sample_width=p.get_sample_size(FORMAT),
        frame_rate=RATE,
        channels=CHANNELS
    )
    audio = edit_audio(sound)
    path = tempdirpath + '\\rt_recorded_audio.wav'
    sf_write(path, audio, 16000, 'PCM_16')
    test_audio = preprocess_dataset([str(path)])
    prediction = modelEN.predict(test_audio.batch(1))
    global STATE
    STATE = "READY"


@multitasking.task
def getInput():
    print("SpeechRecognition")
    global MODE
    global LANGUAGE
    global IS_ACTIVE
    global SILENCE_RMS
    global SILENCE_dB
    global STATE
    while True:
        x = input()
        if x == "ACTIVE_OFF":
            IS_ACTIVE = False
        elif x == "ACTIVE_ON":
            IS_ACTIVE = True
        elif x == "LANGUAGE_PL":
            LANGUAGE = "PL"
        elif x == "LANGUAGE_EN":
            LANGUAGE = "EN"
        elif x == "GET_STATE":
            print("STATE:", STATE)
        elif x=="SET_SILENCE":
            get_silence_db()
        elif x=="GET_SILENCE":
            print("SILENCE_dB:", SILENCE_dB)
            print("SILENCE_RMS:", SILENCE_RMS)
        elif "SET_dB:" in x:
            SILENCE_dB = float(x[7:])
        elif "SET_RMS:" in x:
            SILENCE_RMS = float(x[8:])




getInput()
STATE = "LOADING"

modelEN = models.load_model(MODEL_EN)

modelPL = models.load_model(MODEL_PL)


p = PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
asrRunner()


@multitasking.task
def predict():
    frames = []
    frames_counter = 0
    recording = False
    ignore=False
    while True:
        if IS_ACTIVE:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow = False), dtype=np.int16)
            frames.append(data)
            if len(frames) > 28:
                frames.pop(0)
            peak = np.average(np.abs(data)) * 2
            rms = audioop.rms(data, 2)
            if rms > SILENCE_RMS and not recording:
                recording = True
            if recording:
                frames_counter += 1
            if rms < SILENCE_RMS and frames_counter > 6:

                if(not ignore):
                    first_index = len(frames) - (frames_counter+3)
                    del frames[0: first_index]
                    sound = AudioSegment(
                        data=b''.join(frames),
                        sample_width=p.get_sample_size(FORMAT),
                        frame_rate=RATE,
                        channels=CHANNELS
                    )
                    frames_counter = 0
                    audio = edit_audio(sound)
                    path = tempdirpath + '\\rt_recorded_audio.wav'
                    sf_write(path, audio, 16000, 'PCM_16')
                    frames = []
                    test_audio = preprocess_dataset([str(path)])
                    try:
                        if(LANGUAGE == "EN"):
                            prediction = modelEN.predict(test_audio.batch(1))
                        elif(LANGUAGE == "PL"):
                            prediction = modelPL.predict(test_audio.batch(1))
                        max_predict = np.argmax(prediction)
                        if (LANGUAGE == "EN"):
                            word = COMMANDS_EN[max_predict]
                        elif (LANGUAGE == "PL"):
                            word = COMMANDS_PL[max_predict]
                        max_pred = tf.nn.softmax(prediction[0])
                        pred = max(max_pred)
                        pred = pred.numpy() * 100
                        timestamp = int(time.time() * 1000.0)
                        print("COMMAND:", word, ":", "{0:.0f}".format(pred), ":", timestamp)
                        sys.stdout.flush()
                        _ = gc.collect()

                    except tf.errors.InvalidArgumentError:
                        pass
                    recording = False
                ignore=False

predict()
