from keras.datasets.fashion_mnist import load_data
from keras.utils import to_categorical
from keras import Sequential, layers
from matplotlib import pyplot as plt

NUMBER_OF_CLASSES = 10
IMG_HEIGHT = 28
IMG_WIDTH = 28

(X_train, y_train), (X_test, y_test) = load_data()

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_train.shape[1],
X_train.shape[2], 1))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUMBER_OF_CLASSES, activation="softmax")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), batch_size=32)

print("Evaluate")
result = model.evaluate(X_test, y_test)

plt.subplot(2,1,1)
plt.title('Accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(2,1,2)
plt.title('Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.suptitle(f"Model accuracy\nTest loss: {result[0]}\nTest accuracy: {result[1]}")
print('Test loss:', result[0])
print('Test accuracy:', result[1])

plt.tight_layout
plt.savefig('Lab05/Z5.png')
plt.show()

