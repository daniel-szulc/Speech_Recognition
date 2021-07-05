import pathlib
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt

data_dir = pathlib.Path('dataset/en')
EPOCHS = 100


def get_commands(path):
    commands = np.array(tf.io.gfile.listdir(str(path)))
    print('Commands:', commands)
    return commands


COMMANDS = get_commands(data_dir)


def prepare_model(spectrogram_ds):
    input_shape = None
    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
    print('Input shape:', input_shape)
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

    early = tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)

    check = tf.keras.callbacks.ModelCheckpoint('.\\trainedModel\\enModel.hdf5', monitor='val_accuracy', verbose=1,
                                               save_best_only=True, save_weights_only=False, mode='auto')
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    model.save('.\\trainedModel\\enModel1.hdf5')
    metrics = history.history
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()


def main():
    train_ds = tf.data.experimental.load(
        "./preparedDataSet/train_ds", element_spec=None, compression=None, reader_func=None
    )
    val_ds = tf.data.experimental.load(
        "./preparedDataSet/val_ds", element_spec=None, compression=None, reader_func=None
    )
    spectrogram_ds = tf.data.experimental.load(
        "./preparedDataSet/spectrogram_ds", element_spec=None, compression=None, reader_func=None
    )

    model = prepare_model(spectrogram_ds)
    train_model(model, val_ds, train_ds)


if __name__ == '__main__':
    main()
