import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Sequential API
# model = keras.Sequential(
#     [
#         keras.Input(shape=(32,32,3)),
#         layers.Conv2D(32,3,padding='valid',activation='relu'),
#         layers.MaxPool2D(pool_size=(2,2)),
#         layers.Conv2D(64,3,activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Conv2D(128,3,activation='relu'),
#         layers.Flatten(),
#         layers.Dense(64,activation='relu'),
#         layers.Dense(10),
#
#     ]
# )

# Functional API
def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = my_model()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Use from_logits=False since softmax is applied
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"],
)

print(model.summary())
model.fit(x_train, y_train, batch_size=64, epochs=200, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
