"""
Bonus - 1D CNN on the Chronic Kidney Disease data (TensorFlow / Keras).

Same architecture and splits as classification/04_deep_cnn/02_deep_cnn_classification.ipynb.
Neural-network training is not bit-for-bit deterministic on every machine, so the
numbers can differ slightly from the report (test accuracy ~0.96-0.99).

Run from the repository root:
    python demo/run_cnn.py
"""
import os
import random
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from run_classification import load_ckd, make_preprocessor

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)


def main():
    df = load_ckd()
    X, y = df.drop(columns="class"), df["class"].map({"ckd": 1, "notckd": 0})
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr_full, y_tr_full, test_size=0.2,
                                                random_state=42, stratify=y_tr_full)
    print(f"CNN fit {len(X_tr)} / validation {len(X_val)} / test {len(X_te)}")

    pre = make_preprocessor().fit(X_tr)                  # fit on CNN training subset only
    def to3d(A):                                         # (n, 34) -> (n, 34, 1) for Conv1D
        M = pre.transform(A)
        M = M.toarray() if hasattr(M, "toarray") else M
        return np.asarray(M, dtype="float32")[..., None]

    A_tr, A_val, A_te = to3d(X_tr), to3d(X_val), to3d(X_te)
    print("Input shape per patient:", A_tr.shape[1:], "(34 features x 1 channel)")

    L = tf.keras.layers
    model = tf.keras.Sequential([
        L.Input(shape=A_tr.shape[1:]),
        L.Conv1D(16, 3, activation="relu", padding="same"), L.MaxPooling1D(2), L.Dropout(0.25),
        L.Conv1D(32, 3, activation="relu", padding="same"), L.MaxPooling1D(2), L.Dropout(0.25),
        L.Flatten(), L.Dense(16, activation="relu"), L.Dropout(0.30),
        L.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    print("Trainable parameters:", model.count_params())

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    hist = model.fit(A_tr, y_tr, validation_data=(A_val, y_val), epochs=100, batch_size=16,
                     callbacks=[es], verbose=0)
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    print(f"Trained {len(hist.history['loss'])} epochs, best val loss at epoch {best_epoch}")

    prob = model.predict(A_te, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"\nTEST  acc={accuracy_score(y_te, pred):.4f}  prec={precision_score(y_te, pred):.4f}  "
          f"recall={recall_score(y_te, pred):.4f}  F1={f1_score(y_te, pred):.4f}  "
          f"AUC={roc_auc_score(y_te, prob):.4f}  | TN={tn} FP={fp} FN={fn} TP={tp}")


if __name__ == "__main__":
    main()
