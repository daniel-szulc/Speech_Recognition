import pathlib

from tensorflow.keras import models
import matplotlib.pyplot as plt
import seaborn as sns
import modelTraining
import tensorflow as tf

import numpy as np
#COMMANDS = ['attack', 'left', 'pause', 'right', 'shoot', 'stop', 'unknown', 'up']
data_dir = pathlib.Path('datasetExit/dataset/en')

MODEL = '.\\trainedModel\\enModel.h5'

#COMMANDS = get_commands(data_dir)

def get_commands(path):
    commands = np.array(tf.io.gfile.listdir(str(path)))
    print('Commands:', commands)
    return commands

def confusion_matrix(test_ds, COMMANDS, MODEL):
    model = models.load_model(MODEL)
    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=COMMANDS, yticklabels=COMMANDS,
                annot=True, fmt='g')
    plt.xlabel('Przewidziana komenda')
    plt.ylabel('Komenda')
    plt.show()


def main():
    global COMMANDS

    COMMANDS = modelTraining.get_commands(modelTraining.data_dir)


if __name__ == '__main__':
    main()
