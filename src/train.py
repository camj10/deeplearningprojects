"""
Utilidades de entrenamiento y callbacks comunes. Tracking opcional con MLflow.
"""
import os, contextlib

def compile_and_train(model, x_train, y_train, x_val, y_val,
                      epochs=10, batch_size=64, optimizer="adam",
                      loss=None, metrics=("accuracy",)):
    import tensorflow as tf
    if loss is None:
        # inferir loss por tipo de salida (clasificación multiclase)
        loss = "sparse_categorical_crossentropy"
    model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics))
    cb = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1)
    ]
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                     epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=2)
    return hist

@contextlib.contextmanager
def maybe_mlflow(run_name="run"):
    try:
        import mlflow
        mlflow.set_experiment("project-ml-portfolio")
        with mlflow.start_run(run_name=run_name):
            yield mlflow
    except Exception:
        # MLflow no disponible: seguir sin tracking
        yield None
