"""
Módulos de carga y preprocesamiento de datos para el portfolio.
"""
from typing import Tuple
import numpy as np

def get_cifar10_tf(normalize: bool = True):
    """Carga CIFAR-10 desde tf.keras.datasets (si está disponible)."""
    try:
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        if normalize:
            x_train = x_train.astype("float32")/255.0
            x_test  = x_test.astype("float32")/255.0
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar CIFAR-10: {e}")

def get_imdb_tf(num_words: int = 20000, maxlen: int = 256):
    """Carga IMDB (clasificación binaria) con padding."""
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test  = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    return (x_train, y_train), (x_test, y_test)

def make_sine_series(n_steps=2000, noise=0.1, seed=42):
    """Serie temporal seno con ruido (para LSTM)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    series = np.sin(0.02*t) + 0.5*np.sin(0.05*t) + noise*rng.standard_normal(n_steps)
    return series.astype("float32")
