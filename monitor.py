import pandas as pd
import numpy as np
import joblib
import time
import os
import psycopg2
from datetime import datetime
from tensorflow.keras.models import load_model

# ── Cargar modelos ──
print("Cargando modelos...")
scaler = joblib.load(r'C:\ids\models\scaler.pkl')
le = joblib.load(r'C:\ids\models\label_encoder.pkl')
rf = joblib.load(r'C:\ids\models\modelo_rf.pkl')
autoencoder = load_model(r'C:\ids\models\modelo_autoencoder.keras')
umbral = np.load(r'C:\ids\models\umbral_autoencoder.npy')[0]
print("✅ Modelos cargados")

# ── Conexión PostgreSQL ──
conn = psycopg2.connect(
    host='localhost',
    dbname='ids_db',
    user='postgres',
    password='1234'  # <-- cambia esto
)
cur = conn.cursor()
print("✅ Conectado a PostgreSQL\n")

RENAME_MAP = {
    'Total Fwd Packet':           'Total Fwd Packets',
    'Total Bwd packets':          'Total Backward Packets',
    'Total Length of Fwd Packet': 'Total Length of Fwd Packets',
    'Total Length of Bwd Packet': 'Total Length of Bwd Packets',
    'Packet Length Min':          'Min Packet Length',
    'Packet Length Max':          'Max Packet Length',
    'CWR Flag Count':             'CWE Flag Count',
    'Fwd Segment Size Avg':       'Avg Fwd Segment Size',
    'Bwd Segment Size Avg':       'Avg Bwd Segment Size',
    'Fwd Bytes/Bulk Avg':         'Fwd Avg Bytes/Bulk',
    'Fwd Packet/Bulk Avg':        'Fwd Avg Packets/Bulk',
    'Fwd Bulk Rate Avg':          'Fwd Avg Bulk Rate',
    'Bwd Bytes/Bulk Avg':         'Bwd Avg Bytes/Bulk',
    'Bwd Packet/Bulk Avg':        'Bwd Avg Packets/Bulk',
    'Bwd Bulk Rate Avg':          'Bwd Avg Bulk Rate',
    'FWD Init Win Bytes':         'Init_Win_bytes_forward',
    'Bwd Init Win Bytes':         'Init_Win_bytes_backward',
    'Fwd Act Data Pkts':          'act_data_pkt_fwd',
    'Fwd Seg Size Min':           'min_seg_size_forward',
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

CARPETA_SALIDA    = r'C:\ids\output'
CARPETA_PROCESADOS = r'C:\ids\procesados'
INTERVALO_SEGUNDOS = 10
os.makedirs(CARPETA_PROCESADOS, exist_ok=True)

def guardar_en_db(timestamp, src_ip, dst_ip, dst_port, prediccion, confianza, anomalia_ae, error_ae):
    try:
        cur.execute("""
            INSERT INTO alertas (timestamp, src_ip, dst_ip, dst_port,
                                 prediccion, confianza, anomalia_ae, error_ae)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (timestamp, src_ip, dst_ip, int(dst_port),
              prediccion, float(confianza), bool(anomalia_ae), float(error_ae)))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  ❌ Error DB: {e}")

def procesar_csv(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = df.columns.str.strip()
        df = df.rename(columns=RENAME_MAP)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if df.empty:
            print(f"  ⚠️  Sin filas válidas en {os.path.basename(filepath)}")
            return

        meta = df[['Src IP', 'Dst IP', 'Dst Port']].reset_index(drop=True)
        X = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        X_final_arr = X_scaled[TOP20].values

        pred = rf.predict(X_final_arr)
        pred_label = le.inverse_transform(pred)
        proba = rf.predict_proba(X_final_arr).max(axis=1)

        reconstruido = autoencoder.predict(X_final_arr, verbose=0)
        error = np.mean(np.power(X_final_arr - reconstruido, 2), axis=1)
        anomalia_ae = (error > umbral).astype(int)

        # Guardar cada flujo en PostgreSQL
        ts = datetime.now()
        for i in range(len(pred_label)):
            guardar_en_db(
                ts,
                meta.iloc[i]['Src IP'],
                meta.iloc[i]['Dst IP'],
                meta.iloc[i]['Dst Port'],
                pred_label[i],
                proba[i],
                bool(anomalia_ae[i]),
                error[i]
            )

        total = len(pred_label)
        ataques = int(np.sum(pred_label != 'BENIGN'))
        anomalias = int(np.sum(anomalia_ae))

        print(f"[{ts.strftime('%H:%M:%S')}] 📄 {os.path.basename(filepath)}")
        print(f"  Flujos: {total} | RF Ataques: {ataques} | AE Anomalías: {anomalias} | 💾 Guardado en DB")

        if ataques > 0:
            print("  🚨 ATAQUES DETECTADOS:")
            for i, label in enumerate(pred_label):
                if label != 'BENIGN':
                    print(f"     {meta.iloc[i]['Src IP']} → {meta.iloc[i]['Dst IP']}:{meta.iloc[i]['Dst Port']} | {label} ({proba[i]:.2f})")

        # Mover a procesados
        nombre = os.path.basename(filepath)
        destino = os.path.join(CARPETA_PROCESADOS, nombre)
        if os.path.exists(destino):
            ts2 = datetime.now().strftime('%H%M%S')
            destino = os.path.join(CARPETA_PROCESADOS, f"{ts2}_{nombre}")
        os.rename(filepath, destino)

    except Exception as e:
        print(f"  ❌ Error: {e}")

# ── Loop principal ──
print(f"Vigilando: {CARPETA_SALIDA}")
print(f"Intervalo: cada {INTERVALO_SEGUNDOS} segundos")
print("─" * 50)

while True:
    try:
        csvs = [f for f in os.listdir(CARPETA_SALIDA) if f.endswith('.csv')]
        for nombre in csvs:
            ruta = os.path.join(CARPETA_SALIDA, nombre)
            time.sleep(2)
            procesar_csv(ruta)
    except KeyboardInterrupt:
        print("\n⛔ Monitor detenido.")
        cur.close()
        conn.close()
        break
    except Exception as e:
        print(f"Error en loop: {e}")
    time.sleep(INTERVALO_SEGUNDOS)