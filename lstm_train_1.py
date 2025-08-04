import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ============ STEP 1: Load Training & Validation Dataset ============

train_path = "training.csv"
val_path = "validation.csv"

df_train = pd.read_csv(train_path, sep='\t')  # atau sep=',' jika kamu pakai koma
df_val = pd.read_csv(val_path, sep='\t')


# ============ STEP 2: Preprocessing ============

# Ambil midi_note dan token_idx untuk masing-masing
X_train_raw = df_train[['midi_note']].values
y_train_raw = df_train['token_idx'].values

X_val_raw = df_val[['midi_note']].values
y_val_raw = df_val['token_idx'].values

# Hitung jumlah kelas
num_classes = len(np.unique(np.concatenate([y_train_raw, y_val_raw])))

# One-hot encode label
y_train_cat = to_categorical(y_train_raw, num_classes=num_classes)
y_val_cat = to_categorical(y_val_raw, num_classes=num_classes)

# Buat fungsi untuk membuat sequence
def create_sequences(X_raw, y_cat, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X_raw) - seq_len):
        X_seq.append(X_raw[i:i+seq_len])
        y_seq.append(y_cat[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

SEQ_LEN = 10
X_train, y_train = create_sequences(X_train_raw, y_train_cat, SEQ_LEN)
X_val, y_val = create_sequences(X_val_raw, y_val_cat, SEQ_LEN)


# ============ STEP 3: Bangun Model LSTM ============

model = Sequential([
    LSTM(128, input_shape=(SEQ_LEN, 1), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ============ STEP 4: Training Model ============

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# ============ STEP 5: Visualisasi Loss & Accuracy ============

plt.figure(figsize=(14, 6))

# Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("output/loss_accuracy_plot.png")
plt.show()

# ============ STEP 6: Evaluasi Model ============

# Prediksi kelas token_idx untuk data validasi
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

# Evaluasi klasifikasi
print("\n[Classification Report]")
print(classification_report(y_true, y_pred))

f1 = f1_score(y_true, y_pred, average='weighted')
print(f"\nWeighted F1-Score: {f1:.4f}")

# Confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, cmap='Blues', annot=False, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig("output/confusion_matrix.png")
plt.show()

# ============ STEP 7: Simpan Model & Label ============

os.makedirs("output", exist_ok=True)
model.save("output/lstm_tablature_model.h5")
print("✅ Model saved to output/lstm_tablature_model.h5")

unique_labels = np.unique(y_raw)
joblib.dump(unique_labels, "output/label_tokens.pkl")
print("✅ Label tokens saved to output/label_tokens.pkl")