"""
Métricas y visualizaciones
"""
from typing import Dict, List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt

# Clasificación (TensorFlow + sklearn)
def classification_report_tf(model,
                             x,
                             y_true,
                             label_names: Optional[List[str]] = None,
                             save_cm_path: Optional[str] = None) -> Dict[str, float]:
    """
    Evalúa un modelo de clasificación (softmax multiclase o sigmoide binaria).
    Devuelve dict con 'accuracy' y 'f1' (macro en multiclase, binary en binaria).
    Si save_cm_path se indica, guarda la matriz de confusión en PNG.
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

    y_prob = model.predict(x, verbose=0)

    if y_prob.ndim == 2 and y_prob.shape[1] > 1:
        # softmax multiclase
        y_pred = np.argmax(y_prob, axis=1).reshape(-1)
        y_true = np.array(y_true).reshape(-1)
        average = "macro"
    else:
        # sigmoide binaria
        y_pred = (np.array(y_prob).ravel() >= 0.5).astype(int)
        y_true = np.array(y_true).reshape(-1)
        average = "binary"

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average=average)

    if save_cm_path is not None:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        fig, ax = plt.subplots(figsize=(4,4))
        disp.plot(ax=ax, values_format="d", colorbar=False)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_cm_path), exist_ok=True)
        plt.savefig(save_cm_path, dpi=150)
        plt.close(fig)

    return {"accuracy": float(acc), "f1": float(f1)}

def count_params(model) -> int:
    try:
        return int(model.count_params())
    except Exception:
        return -1

def save_history_curves(history, out_png: str) -> None:
    """Guarda curvas de entrenamiento (loss/acc y val_*) en un PNG."""
    h = getattr(history, "history", None)
    if not h:
        return
    plt.figure()
    if "loss" in h: plt.plot(h["loss"], label="loss")
    if "val_loss" in h: plt.plot(h["val_loss"], label="val_loss")
    if "accuracy" in h: plt.plot(h["accuracy"], label="acc")
    if "val_accuracy" in h: plt.plot(h["val_accuracy"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("metric"); plt.title("Learning curves"); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


# GAN: FID 
def compute_fid_tf(real_images: np.ndarray,
                   gen_images: np.ndarray,
                   batch_size: int = 64) -> float:
    """
    Calcula FID entre real_images y gen_images usando InceptionV3 de Keras.
    real_images y gen_images pueden estar en [0,1] o [0,255]; se ajustan internamente.
    Requiere scipy y TensorFlow.
    """
    import tensorflow as tf
    # from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

    # from tensorflow.keras.applications import InceptionV3
    # from tensorflow.keras.applications.inception_v3 import preprocess_input



    from keras.applications.inception_v3 import InceptionV3, preprocess_input



    from scipy.linalg import sqrtm

    def _prep(x: np.ndarray) -> np.ndarray:
        x = x.astype("float32")
        if x.max() <= 1.1:
            x = x * 255.0
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        return preprocess_input(x)

    inc = InceptionV3(include_top=False, pooling="avg", input_shape=(299,299,3))

    def _embeddings(images: np.ndarray) -> np.ndarray:
        ims = tf.image.resize(images, (299, 299)).numpy()
        ims = _prep(ims)
        outs = []
        for i in range(0, len(ims), batch_size):
            outs.append(inc.predict(ims[i:i+batch_size], verbose=0))
        return np.concatenate(outs, axis=0)

    act1 = _embeddings(real_images)
    act2 = _embeddings(gen_images)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

# Imágenes 
def save_image_grid(images: np.ndarray, out_png: str, ncols: int = 4) -> None:
    """Guarda una grilla de imágenes en PNG."""
    n = len(images)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.ravel() if n > 1 else [axes]
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
