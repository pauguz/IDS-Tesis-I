import joblib
import pandas as pd
import numpy as np

scaler = joblib.load(r'C:\ids\models\scaler.pkl')
df = pd.read_csv(r'C:\ids\output\prueba_pcap_Flow.csv', low_memory=False)
df.columns = df.columns.str.strip()

print("=== COLUMNAS EN CSV ===")
for c in df.columns:
    print(c)

print("\n=== COLUMNAS QUE ESPERA EL SCALER ===")
for c in scaler.feature_names_in_:
    print(c)