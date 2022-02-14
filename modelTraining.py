import pathlib
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

import recognitionTest

model_language ='en' #'pl'

model_location = '.\\trainedModel\\'+model_language+'Model.h5'
best_model_location = '.\\trainedModel\\best_model_'+model_language+'.h5'

data_dir = pathlib.Path('datasetExit/dataset/' + model_language)
EPOCHS = 1000


def get_commands(path):
    commands = np.array(tf.io.gfile.listdir(str(path)))
    print('Commands:', commands)
    return commands


COMMANDS = get_commands(data_dir)

def prepare_model(spectrogram_ds):
    input_shape = None
    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape

    num_labels = len(COMMANDS)

    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()
    return model


def train_model(model, val_ds, train_ds):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )



    early = tf.keras.callbacks.EarlyStopping(verbose=1, patience=200)
    check = tf.keras.callbacks.ModelCheckpoint(model_location, monitor='val_loss', mode='min',
                                               verbose=1, save_best_only=True)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early, check]
    )
    metrics = history.history
    train_metrics = history.history['val_accuracy']
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, history.history['val_accuracy'])
    plt.plot(epochs, history.history['accuracy'])
    plt.plot(epochs, history.history['val_loss'])
    plt.plot(epochs, history.history['loss'])
    plt.xlabel("Epoki")
    plt.ylabel("Dokładność/Strata")
    plt.legend(['dokładność zbioru walidacyjnego (val_accuracy)', 'dokładność zbioru uczącego (accuracy)', 'strata zbioru walidacyjnego (val_loss)', 'strata zbioru uczącego (loss)'])
    ax = plt.gca()
    ax.set_ylim([0, 1.1])
    plt.show()

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


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
    val_files = filenames[num_80_samples: num_80_samples + num_10_samples]
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


    rows = 2
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 7))
    for i, (audio, label) in enumerate(waveform_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(audio.numpy())
        ax.set_yticks(np.arange(-1.0, 1.2, 0.2))
        label = label.numpy().decode('utf-8')
        ax.set_title(label)
    plt.show()

    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)
    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, spectrogram_ds


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds



def get_spectrogram(waveform):

    zero_padding = tf.zeros(16000 - tf.shape(waveform), dtype=tf.float32)

    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)


    return spectrogram


def plot_spectrogram(spectrogram, ax):

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



def main():
    train_ds, val_ds, test_ds, spectrogram_ds = prepare_training_dataset(data_dir)
    model = prepare_model(spectrogram_ds)
    plot_model(model, show_shapes=True, show_layer_names=True)
    train_model(model, val_ds, train_ds)

    recognitionTest.confusion_matrix(test_ds, COMMANDS, model_location)
    recognitionTest.confusion_matrix(test_ds, COMMANDS, best_model_location)



if __name__ == '__main__':
    main()
