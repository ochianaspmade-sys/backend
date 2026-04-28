import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# --- 1. SETUP PATH & LOAD MODEL ---
base_path = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(base_path, 'model_hipertensi_xgb_pso.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'scaler_hipertensi.pkl'))
    print("✅ Berhasil: Model dan Scaler siap!")
except Exception as e:
    print(f"❌ Gagal: {e}")

# --- 2. LOGIKA PAKAR ---
def klasifikasi_jnc7_text(sistole, diastole):
    sys, dia = float(sistole), float(diastole)
    if sys >= 160 or dia >= 100: return "Hypertension Stage 2"
    if (140 <= sys <= 159) or (90 <= dia <= 99): return "Hypertension Stage 1"
    if (120 <= sys <= 139) or (80 <= dia <= 89): return "Pre-Hypertension"
    return "Normal"

def hitung_expert_score(data_json):
    score = 0
    sys, dia = float(data_json['sistole']), float(data_json['diastole'])
    age, imt = float(data_json['umur']), float(data_json['imt'])
    merokok = int(data_json['merokok'])
    if sys >= 160 or dia >= 100: score += 5
    elif sys >= 140 or dia >= 90: score += 3
    elif sys >= 120 or dia >= 80: score += 1
    if age >= 60: score += 1
    if imt >= 30: score += 1
    if merokok == 1: score += 1
    return score

# --- 3. ROUTE PREDIKSI ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        sistole = float(data['sistole'])
        diastole = float(data['diastole'])
        bmi_value = float(data['imt'])
        imt_cat = int(data.get('imt_category', 1))

        # 1. Hitung Skor Pakar & Klasifikasi JNC
        expert_score = hitung_expert_score(data)
        jnc_class = klasifikasi_jnc7_text(sistole, diastole)

        # 2. LOGIKA HYBRID (JARING PENGAMAN)
        if sistole >= 160 or diastole >= 100:
            # JALUR A: SISTEM PAKAR MEMOTONG JALUR (Kondisi Mutlak)
            kesimpulan = "TERDETEKSI HYPERTENSION STAGE 2"
            probabilitas = 100.0
            sumber = "Sistem Pakar (Aturan Mutlak JNC)"
            is_high_risk = True
        else:
            # JALUR B: MODEL XGBOOST-PSO MENGAMBIL ALIH (Area Abu-abu)
            
            # A. Urutan Scaler (8 Kolom Numerik Sesuai Colab)
            data_untuk_scaler = np.array([[
                float(data['umur']), float(data['tinggi']), float(data['berat']),
                sistole, diastole,
                bmi_value, float(imt_cat), float(expert_score)
            ]])
            features_scaled = scaler.transform(data_untuk_scaler)

            # B. Urutan Model (10 Kolom: JK & Merokok di depan)
            final_features = np.hstack([
                np.array([[int(data['jenis_kelamin']), int(data['merokok'])]]), 
                features_scaled
            ])

            # C. Eksekusi Prediksi
            prediction_idx = int(model.predict(final_features)[0])
            probabilities = model.predict_proba(final_features)[0]
            probabilitas = float(np.max(probabilities) * 100)

            # Mapping Label XGBoost
            map_labels = {0: "Normal", 1: "Pre-Hypertension", 2: "Hypertension Stage 1", 3: "Hypertension Stage 2"}
            prediction_ai = map_labels.get(prediction_idx, "Unknown")

            if prediction_idx == 0:
                kesimpulan = "NORMAL (TIDAK TERDETEKSI HIPERTENSI)"
                is_high_risk = False
            else:
                kesimpulan = f"TERDETEKSI {prediction_ai.upper()}"
                is_high_risk = True
                
            sumber = "Model AI (XGBoost-PSO)"

        # 3. PEMBUNGKUSAN DATA KE FRONTEND
        return jsonify({
            "status": "success",
            "data": {
                "tekanan_darah": f"{int(sistole)}/{int(diastole)} mmHg",
                "klasifikasi_jnc": jnc_class,
                "skor_pakar": f"{expert_score} poin",
                "kesimpulan": kesimpulan,
                "probabilitas": f"{probabilitas:.2f}",
                "sumber": sumber,
                "is_high_risk": is_high_risk,
                "input_data": {
                    "sistole": sistole,
                    "diastole": diastole,
                    "imt": bmi_value,
                    "umur": data.get('umur', 0),
                    "imt_category": imt_cat
                }
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
