
import keras
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers.core import Dense, Activation, Dropout
from keras import regularizers
from tensorflow.keras.callbacks import History 
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)     
X_test = X_test.reshape(10000, 784)
## normalize the data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

train_set, valid_set, train_labels_set, valid_labels_set = train_test_split(X_train, Y_train, test_size=0.1, shuffle= True)

epochs = 100

model = Sequential()    
model.add(Dense(100, kernel_regularizer=regularizers.l2(0.0001))) 
model.add(Activation('tanh'))
model.add(Dense(150, kernel_regularizer=regularizers.l2(0.0001))) 
model.add(Activation('tanh'))
model.add(Dense(10)) 
model.add(Activation('softmax'))

sgd = optimizers.SGD(learning_rate=0.01, momentum = 0.01, nesterov=False)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
history = History()
history = model.fit(train_set, train_labels_set, 
            epochs=epochs, 
            verbose=1, 
            validation_data=(valid_set, valid_labels_set),
            callbacks=[history])

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy:', score[1])
print('Test Score:', score[0])

# Plot training and validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# plot history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Categorical Crossentropy loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()