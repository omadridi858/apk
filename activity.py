import numpy as np
# from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical

from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
def start(o, X_train, Y_train, test_images, test_labels,cl,self):
    global history, model

    if o == 1:
        fig1=cnn(X_train, Y_train, test_images, test_labels,cl)
        return fig1
    elif o == 2:
        fig1=MLP(X_train, Y_train, test_images, test_labels,cl)
        return fig1
    elif o == 3:
        try:
            fig2=testcnn(X_train, Y_train, test_images, test_labels,cl)
            return fig2
        except:
            print('somthing is not true') 
            pass
    elif o == 4:
        try:
            fig2=testmlp(X_train, Y_train, test_images, test_labels,cl)
            return fig2
        except:
            print('somthing is not true')
            pass
    else:
        exit()



def cnn(X_train, Y_train, test_images, test_labels,cl):
    global model, history
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(11, activation='softmax')

    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history1 = model.fit(X_train, Y_train, batch_size=400, epochs=6, validation_split=0.1)
    # Save and reload the model
    model.save('CNN.h5')
    return history1
def testcnn(X_train, Y_train, test_images, test_labels,cl):
    model = load_model('CNN.h5')
    test_loss = model.evaluate(test_images, test_labels)

    predictions = model.predict(test_images)
    pred_labels = np.argmax(predictions, axis=1)

    class_names = cl
    CM = confusion_matrix(test_labels, pred_labels)
    
    history2=[CM,class_names]
    return history2




def MLP(X_train, Y_train, test_images, test_labels,cl):
    print("Shape of training data:")
    print(X_train.shape)  # Print the shape to debug
    print(Y_train.shape)  # Print the shape to debug

    print("Shape of test data:")
    print(test_images.shape)  # Print the shape to debug
    print(test_labels.shape)  # Print the shape to debug

    Y_train = to_categorical(Y_train, num_classes=11)
    test_labels = to_categorical(test_labels, num_classes=11)

    # Calculate the number of features in the input data
    num_features = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]

    # Reshape the input data
    X_train = X_train.reshape(X_train.shape[0], num_features)
    test_images = test_images.reshape(test_images.shape[0], num_features)

    X_train = X_train.astype('float32')
    test_images = test_images.astype('float32')

    X_train /= 255
    test_images /= 255

    # Define and compile the model
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=num_features))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(11, activation='softmax'))
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history1 = model.fit(X_train, Y_train, epochs=15, batch_size=32, verbose=2, validation_split=0.2)

    # Plot losses and accuracies

    # Save and reload the model
    model.save('MLP.h5')
    return history1
def testmlp(X_train, Y_train, test_images, test_labels,cl):
    model = load_model('MLP.h5')

    # Evaluate the model on test data
    score = model.evaluate(test_images, test_labels, batch_size=128, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Generate predictions for test samples
    predictions = model.predict(test_images)
    print("Predictions shape:", predictions.shape)

    labels_prediction = np.argmax(predictions, axis=1)
    print(labels_prediction)

    labels = cl
    cm = confusion_matrix(np.argmax(test_labels, axis=1), labels_prediction)
    
    history2=[cm,labels]
    return history2

