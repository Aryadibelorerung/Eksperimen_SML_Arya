import pandas as pd
import requests
import json

# Baca dataset
df = pd.read_csv("titanic_clean.csv")

# Hapus kolom target jika ada
if 'Survived' in df.columns:
    df = df.drop(columns=["Survived"])

# Ambil 1 sampel data secara acak
sample = df.sample(n=1, random_state=None)

# Ubah sampel menjadi format split untuk JSON
data_split = sample.to_dict(orient="split")

# Kirim request POST ke endpoint model
response = requests.post(
    url="http://localhost:8000/predict",
    headers={"Content-Type": "application/json"},
    data=json.dumps({"dataframe_split": data_split})
)

# Tampilkan data yang digunakan untuk inferensi
print("Data digunakan untuk inferensi:")
print(sample)

# Tampilkan respons dari API
print("\nStatus Code:", response.status_code)
try:
    print("Response:", response.json())
except Exception:
    print("Raw Response:", response.text)