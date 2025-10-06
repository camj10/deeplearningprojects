"""
Modelos base: CNN simple, DCGAN (generator/discriminator), LSTM, Transformer (HF).
"""
import tensorflow as tf
from tensorflow.keras import layers

#  CNN para CIFAR-10 
def build_cnn(input_shape=(32,32,3), n_classes=10):
    model = tf.keras.Sequential([
        layers.Conv2D(32,3,activation="relu",padding="same",input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,activation="relu",padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation="relu",padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128,activation="relu"),
        layers.Dense(n_classes,activation="softmax")
    ])
    return model

#  DCGAN (generator / discriminator) 
def make_generator(latent_dim=100):
    return tf.keras.Sequential([
        layers.Input((latent_dim,)),
        layers.Dense(8*8*256),
        layers.Reshape((8,8,256)),
        layers.Conv2DTranspose(128,4,strides=2,padding="same",activation="relu"),
        layers.Conv2DTranspose(64,4,strides=2,padding="same",activation="relu"),
        layers.Conv2DTranspose(3,3,activation="tanh",padding="same") # salida [-1,1]
    ], name="generator")

def make_discriminator(input_shape=(32,32,3)):
    return tf.keras.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(64,4,strides=2,padding="same"), layers.LeakyReLU(0.2),
        layers.Conv2D(128,4,strides=2,padding="same"), layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ], name="discriminator")

#  LSTM para serie temporal (predicción next-step)
def build_lstm(input_timesteps=32, features=1):
    model = tf.keras.Sequential([
        layers.Input((input_timesteps, features)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model
