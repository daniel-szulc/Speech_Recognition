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


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def edit_audio(audio):
    audio = audio.set_channels(1)  # to mono audio
    normalized_sound = effects.normalize(audio)  # normalize volume of audio
    y = np.array(normalized_sound.get_array_of_samples()).astype(np.float32)
    y = y / (1 << 8 * 2 - 1)  # convert to librosa
    new_audio, index = librosa.effects.trim(y, top_db=20)  # cut silence
    duration = librosa.get_duration(new_audio, 16000)
    if duration > 0.99:
        new_audio = librosa.effects.time_stretch(new_audio, duration+0.01)
    return new_audio


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
            new_audio = edit_audio(raw_sound)
            sf.write(new_path, new_audio, 16000, 'PCM_16')


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros(16000 - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


def plot_spectrogram(spectrogram, ax):
    # Convert to frequencies to log scale and transpose so that the time is
    # represented in the x-axis (columns).
    with np.errstate(divide='ignore'):
        log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    x = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    y = range(height)
    ax.pcolormesh(x, y, log_spec, shading='auto')
    return ax


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == COMMANDS)
    return spectrogram, label_id


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds


def prepare_training_dataset(data_directory):
    global COMMANDS
    commands = get_commands(data_directory)
    filenames = tf.io.gfile.glob(str(data_directory) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    print('Number of examples per label:',
          len(tf.io.gfile.listdir(str(data_directory / commands[0]))))
    print('Example file tensor:', filenames[0])
    num_80_samples = int(num_samples * 0.8)  # 80% samples for training
    num_10_samples = int(num_samples * 0.1)  # 10% samples for validation and 10% for tests
    train_files = filenames[:num_80_samples]
    val_files = filenames[num_10_samples: num_10_samples + num_10_samples]
    test_files = filenames[-num_10_samples:]

    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))
    global AUTOTUNE
    AUTOTUNE = tf.data.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)

    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    spectrogram_ds = waveform_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)

    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, spectrogram_ds


def main():
    from_dir = '.\\dataset\\en'
    to_dir = '.\\new_dataset\\'
    prepare_audio_files(from_dir, to_dir)
    data_dir = pathlib.Path(to_dir + from_dir[3:])

    train_ds, val_ds, test_ds, spectrogram_ds = prepare_training_dataset(data_dir)

    tf.data.experimental.save(train_ds, "./preparedDataSet/train_ds", compression=None, shard_func=None)
    tf.data.experimental.save(val_ds, "./preparedDataSet/val_ds", compression=None, shard_func=None)
    tf.data.experimental.save(test_ds, "./preparedDataSet/test_ds", compression=None, shard_func=None)
    tf.data.experimental.save(spectrogram_ds, "./preparedDataSet/spectrogram_ds", compression=None, shard_func=None)
    return


if __name__ == '__main__':
    main()
