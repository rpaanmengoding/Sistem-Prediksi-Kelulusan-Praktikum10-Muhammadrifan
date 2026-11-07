
import joblib
from sklearn.tree import DecisionTreeClassifier
import numpy as np

print("Sedang melatih model AI...")

# 1. Data latih awal: [IPK, Total SKS]
X_awal = np.array([
    [3.5, 144],
    [2.0, 100],
    [3.2, 140],
    [2.5, 110],
    [3.8, 150]
])

# 2. Membuat tambahan 145 data training
np.random.seed(42)

ipk = np.round(
    np.random.uniform(1.0, 4.0, 145), 2
)

sks = np.random.randint(
    60, 161, 145
)

X_tambahan = np.column_stack((ipk, sks))

# 3. Menggabungkan 150 data training
X_train = np.vstack((X_awal, X_tambahan))

# 4. Membuat label berdasarkan aturan simulasi
# IPK minimal 2.00 dan SKS minimal 120

y_train = np.where(
    (X_train[:, 0] >= 2.0) &
    (X_train[:, 1] >= 120),
    1,
    0
)

# 5. Membuat model Decision Tree
model = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

# 6. Melatih model
model.fit(X_train, y_train)

# 7. Menguji beberapa data
print("\n===== HASIL PENGUJIAN MODEL =====")

data_uji = np.array([
    [1.0, 140],
    [1.5, 150],
    [2.0, 100],
    [2.5, 130],
    [3.0, 144],
    [3.8, 150]
])

hasil = model.predict(data_uji)

for data, prediksi in zip(data_uji, hasil):
    status = "LULUS" if prediksi == 1 else "TIDAK LULUS"

    print(
        f"IPK: {data[0]} | "
        f"SKS: {data[1]} | "
        f"Prediksi: {status}"
    )

# 8. Informasi jumlah data
print("\n===== INFORMASI DATA TRAINING =====")
print("Jumlah data training:", len(X_train))
print("Jumlah data Lulus:", np.sum(y_train == 1))
print("Jumlah data Tidak Lulus:", np.sum(y_train == 0))

# 9. Simpan model
joblib.dump(model, 'model_kelulusan.pkl')

print("\nSukses! Model disimpan sebagai 'model_kelulusan.pkl'")
