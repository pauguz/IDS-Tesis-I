import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ── Cargar modelos ──
print("Cargando modelos...")
scaler = joblib.load(r'C:\ids\models\scaler.pkl')
le = joblib.load(r'C:\ids\models\label_encoder.pkl')
rf = joblib.load(r'C:\ids\models\modelo_rf.pkl')
autoencoder = load_model(r'C:\ids\models\modelo_autoencoder.keras')
umbral = np.load(r'C:\ids\models\umbral_autoencoder.npy')[0]
print("✅ Modelos cargados")

# ── Mapeo COMPLETO CICFlowMeter v4 -> nombres de entrenamiento ──
RENAME_MAP = {
    'Total Fwd Packet':       'Total Fwd Packets',
    'Total Bwd packets':      'Total Backward Packets',
    'Total Length of Fwd Packet': 'Total Length of Fwd Packets',
    'Total Length of Bwd Packet': 'Total Length of Bwd Packets',
    'Packet Length Min':      'Min Packet Length',
    'Packet Length Max':      'Max Packet Length',
    'CWR Flag Count':         'CWE Flag Count',
    'Fwd Segment Size Avg':   'Avg Fwd Segment Size',
    'Bwd Segment Size Avg':   'Avg Bwd Segment Size',
    'Fwd Bytes/Bulk Avg':     'Fwd Avg Bytes/Bulk',
    'Fwd Packet/Bulk Avg':    'Fwd Avg Packets/Bulk',
    'Fwd Bulk Rate Avg':      'Fwd Avg Bulk Rate',
    'Bwd Bytes/Bulk Avg':     'Bwd Avg Bytes/Bulk',
    'Bwd Packet/Bulk Avg':    'Bwd Avg Packets/Bulk',
    'Bwd Bulk Rate Avg':      'Bwd Avg Bulk Rate',
    'FWD Init Win Bytes':     'Init_Win_bytes_forward',
    'Bwd Init Win Bytes':     'Init_Win_bytes_backward',
    'Fwd Act Data Pkts':      'act_data_pkt_fwd',
    'Fwd Seg Size Min':       'min_seg_size_forward',
}

TOP20 = [
    'Bwd Packet Length Max', 'Fwd Packet Length Max', 'Init_Win_bytes_forward',
    'Subflow Fwd Bytes', 'Avg Bwd Segment Size', 'act_data_pkt_fwd',
    'Total Length of Bwd Packets', 'Fwd Packet Length Mean', 'Bwd Header Length',
    'Bwd Packets/s', 'Fwd IAT Std', 'Total Length of Fwd Packets',
    'Fwd IAT Total', 'Packet Length Variance', 'Avg Fwd Segment Size',
    'Fwd Header Length', 'Fwd IAT Mean', 'Bwd Packet Length Mean',
    'Max Packet Length', 'Average Packet Size'
]

def procesar_csv(filepath):
    print(f"\nProcesando: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip()

    # Renombrar columnas
    df = df.rename(columns=RENAME_MAP)

    # Limpiar
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if df.empty:
        print("No hay filas válidas.")
        return

    # Guardar metadatos
    meta = df[['Src IP', 'Dst IP', 'Dst Port']].reset_index(drop=True)

    # Alinear columnas con el scaler (rellena con 0 las que falten)
    X = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X_final = X_scaled[TOP20]

    # ── Random Forest ──
    # Convertir a numpy para evitar validación de nombres de columnas
    X_final_arr = X_final.values
    pred = rf.predict(X_final_arr)
    pred_label = le.inverse_transform(pred)
    proba = rf.predict_proba(X_final_arr).max(axis=1)

    # ── Autoencoder ──
    reconstruido = autoencoder.predict(X_final_arr, verbose=0)
    error = np.mean(np.power(X_final_arr - reconstruido, 2), axis=1)
    anomalia_ae = (error > umbral).astype(int)



    # ── Mostrar resultados ──
    resultados = pd.DataFrame({
        'Src IP':        meta['Src IP'],
        'Dst IP':        meta['Dst IP'],
        'Dst Port':      meta['Dst Port'],
        'Prediccion_RF': pred_label,
        'Confianza':     proba.round(3),
        'Anomalia_AE':   anomalia_ae,
        'Error_AE':      error.round(6)
    })

    print(f"\nTotal flujos procesados: {len(resultados)}")
    print(resultados.to_string(index=False))

    print("\n── Resumen ──")
    print(resultados['Prediccion_RF'].value_counts())
    print(f"Anomalías AE detectadas: {resultados['Anomalia_AE'].sum()}")

if __name__ == '__main__':
    procesar_csv(r'C:\ids\output\prueba_pcap_Flow.csv')