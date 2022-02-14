import os
import pathlib
from tqdm import tqdm
import soundfile as sf
import librosa
import tensorflow as tf
import numpy as np
from pydub import AudioSegment, effects

COMMANDS = None
AUTOTUNE = None


def get_commands(path):
    commands = np.array(tf.io.gfile.listdir(str(path)))
    print('Commands:', commands)
    return commands


def edit_audio(audio):
    audio = audio.set_channels(1)  # to mono audio
    normalized_sound = effects.normalize(audio)  # normalize volume of audio
    y = np.array(normalized_sound.get_array_of_samples()).astype(np.float32)
    y = y / (1 << 8 * 2 - 1)  # convert to librosa
    new_audio, index = librosa.effects.trim(y, top_db=30)  # cut silence
    duration = librosa.get_duration(new_audio, 16000)
    if duration > 0.99:
        new_audio = librosa.effects.time_stretch(new_audio, duration+0.01)

    return new_audio

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):

    trim_ms = 0

    assert chunk_size > 0
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def prepare_audio_files(from_data_path, to_data_path):
    for soundClass in os.listdir(from_data_path):
        path_sound = from_data_path + '\\' + soundClass
        new_dir = pathlib.Path(to_data_path + path_sound)
        if not new_dir.exists():
            os.makedirs(new_dir)
        for soundFile in tqdm(os.listdir(path_sound), "Converting : '{}'".format(soundClass)):
            img_sound_path = path_sound + '\\' + soundFile
            new_path = to_data_path + img_sound_path[1:]
            raw_sound = AudioSegment.from_file(img_sound_path, "wav")
            start_trim = detect_leading_silence(raw_sound) # cut silence
            end_trim = detect_leading_silence(raw_sound.reverse())
            duration = len(raw_sound)
            trimmed_sound = raw_sound[start_trim:duration - end_trim] # cut silence
            raw_sound = trimmed_sound.set_frame_rate(16000)
            new_audio = edit_audio(raw_sound)
            sf.write(new_path, new_audio, 16000, 'PCM_16')


def main():
    from_dir = '.\\dataset\\en'
    to_dir = '.\\datasetExit'
    prepare_audio_files(from_dir, to_dir)
    return


if __name__ == '__main__':
    main()
