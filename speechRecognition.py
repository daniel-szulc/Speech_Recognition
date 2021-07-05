import pyaudio
import soundfile as sf
import librosa
import time
import tensorflow as tf
from tensorflow.keras import models
from pydub import AudioSegment, effects
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

start_time = 0
MODEL = '.\\trainedModel\\enModel.hdf5'
COMMANDS = ['down', 'forward', 'left', 'right', 'stop']
AUTOTUNE = tf.data.AUTOTUNE
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_TIME = 2
SILENCE_dB = 20


def get_spectrogram(audio):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros(16000 - tf.shape(audio), dtype=tf.float32)
    # Concatenate audio with padding so that all audio clips will be of the
    # same length
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
    test = librosa.core.power_to_db(y)
    print(test)
    print(test.mean())
    y = y / (1 << 8 * 2 - 1)  # convert to librosa
    new_audio, index = librosa.effects.trim(y, top_db=SILENCE_dB)  # cut silence
    duration = librosa.get_duration(new_audio, RATE)
    if duration > 0.99:
        new_audio = librosa.effects.time_stretch(new_audio, duration+0.01)
    return new_audio


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def record_audio_sample():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Record will start")
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    frames = []
    for _ in tqdm(range(0, int(RATE / CHUNK * RECORD_TIME)), "Recording..."):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    global start_time
    start_time = time.time()
    sound = AudioSegment(
        data=b''.join(frames),
        sample_width=p.get_sample_size(FORMAT),
        frame_rate=RATE,
        channels=CHANNELS
    )
    audio = edit_audio(sound)
    path = './testData/recorded_audio.wav'
    sf.write(path, audio, 16000, 'PCM_16')
    return path


def main():
    model = models.load_model(MODEL)
    prediction = None
    recorded_audio = record_audio_sample()
    test_audio = preprocess_dataset([str(recorded_audio)])
    for spectrogram in test_audio.batch(1):
        prediction = model.predict(spectrogram)
        plt.bar(COMMANDS, tf.nn.softmax(prediction[0]) * 100)
        plt.xlabel('commands')
        plt.ylabel('confidence [%]')
    max_predict = np.argmax(prediction)
    word = COMMANDS[max_predict]
    plt.title('Predictions for word: ' + word)
    plt.show()
    end_time = time.time()
    print('recognized word: ' + word)
    print('Time spent for prediction : ' + "{0:.2f}secs".format(end_time - start_time))


if __name__ == '__main__':
    main()
