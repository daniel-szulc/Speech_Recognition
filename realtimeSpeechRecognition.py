import statistics
import warnings
import pyaudio
import soundfile as sf
import librosa
import tensorflow as tf
from tensorflow.keras import models
from pydub import AudioSegment, effects
import numpy as np

MODEL = '.\\trainedModel\\enModel.hdf5'
COMMANDS = ['down', 'forward', 'left', 'right', 'stop']
AUTOTUNE = tf.data.AUTOTUNE
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
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
    y = y / (1 << 8 * 2 - 1)  # convert to librosa
    new_audio, index = librosa.effects.trim(y, top_db=SILENCE_dB)  # cut silence
    duration = librosa.get_duration(new_audio, RATE)
    if duration > 0.99:
        new_audio = librosa.effects.time_stretch(new_audio, duration+0.01)
    return new_audio


def get_silence_db(stream):
    frames = []
    peaks = []
    print('PLEASE BE QUIET...')
    for _ in range(0, int(RATE / CHUNK * 1)):
        data = stream.read(CHUNK)
        frames.append(data)
        data1 = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        peak = np.average(np.abs(data1)) * 2
        peaks.append(peak)
    print('DONE.')
    sound = AudioSegment(
        data=b''.join(frames),
        sample_width=2,
        frame_rate=RATE,
        channels=CHANNELS
    )
    y = np.array(sound.get_array_of_samples()).astype(np.float32)
    array_of_db = librosa.core.power_to_db(y)
    if statistics.mean(peaks) > 1000:
        print("It's too loud!")
    global SILENCE_dB
    SILENCE_dB = abs(array_of_db.mean())*1.1


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def main():
    model = models.load_model(MODEL)
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    get_silence_db(stream)

    print('silence dB: ', SILENCE_dB)
    frames = []
    frames_counter = 0
    recording = False
    print('START RECOGNITION')
    print('SPEAK...')
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        frames.append(data)
        if len(frames) > 28:
            frames.pop(0)
        peak = np.average(np.abs(data)) * 2
        if peak > 2900 and not recording:
            recording = True
        if recording:
            frames_counter += 1
        if peak < 2900 and frames_counter > 6:
            first_index = len(frames) - frames_counter
            del frames[0: first_index]
            sound = AudioSegment(
                data=b''.join(frames),
                sample_width=p.get_sample_size(FORMAT),
                frame_rate=RATE,
                channels=CHANNELS
            )
            frames_counter = 0
            audio = edit_audio(sound)
            path = './testData/rt_recorded_audio.wav'
            sf.write(path, audio, 16000, 'PCM_16')
            frames = []
            test_audio = preprocess_dataset([str(path)])
            try:
                prediction = model.predict(test_audio.batch(1))
                max_predict = np.argmax(prediction)
                word = COMMANDS[max_predict]
                max_pred = tf.nn.softmax(prediction[0])
                pred = max(max_pred)
                pred = pred.numpy() * 100
                print(word + ' : ' + "{0:.2f}%".format(pred))
            except tf.errors.InvalidArgumentError:
                warnings.warn("Too long speech. Should be less than 1 second.")
                pass
            recording = False


if __name__ == '__main__':
    main()
