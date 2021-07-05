from tensorflow.keras import models
import matplotlib.pyplot as plt
import seaborn as sns
import modelTraining
import tensorflow as tf
import numpy as np
COMMANDS = ['down' 'forward' 'left' 'right' 'stop']
MODEL = '.\\trainedModel\\enModel1.hdf5'
model = models.load_model(MODEL)


def confusion_matrix():
    test_ds = tf.data.experimental.load(
        "./preparedDataSet/test_ds", element_spec=None, compression=None, reader_func=None
    )

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
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


def main():
    global COMMANDS
    COMMANDS = modelTraining.get_commands(modelTraining.data_dir)

    confusion_matrix()


if __name__ == '__main__':
    main()
