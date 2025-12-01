import os
import re
import string
import logging
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "all-MiniLM-L6-v2"
INTENT_THRESHOLD = 0.50
FACTORY_API_BASE = "http://127.0.0.1:5000"

# ⚠️ UPDATE TO YOUR EXACT PATH
MODEL_DIR = r'C:\Users\Chirag B A\OneDrive\Desktop\Python Scripts for model\models' 

# CALIBRATION FACTOR (Scales Model Output -> Real World)
CALIBRATION_FACTOR = 5.5 

# ==========================================
# 1. DIGITAL TWIN CLASS (Prediction Engine)
# ==========================================
class FactoryDigitalTwin:
    def __init__(self, model_dir):
        self.loaded = False
        self.power_history = []
        try:
            logging.info(f"Loading models from {model_dir}...")
            self.model_solar = joblib.load(os.path.join(model_dir, 'solar_model.pkl'))
            self.model_load = joblib.load(os.path.join(model_dir, 'consumption_model.pkl'))
            self.encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
            self.loaded = True
            logging.info("✅ Digital Twin Models Loaded Successfully.")
        except Exception as e:
            logging.error(f"❌ Failed to load models: {e}")
            self.loaded = False

    def _encode_value(self, col, val):
        if not self.loaded or col not in self.encoders:
            return 0
        try:
            return self.encoders[col].transform([val])[0]
        except:
            return 0

    def predict_net_grid_load(self, minutes_ahead, user_overrides=None, live_devices=None):
        """
        Main Prediction Logic with:
        1. Live State Parsing
        2. Smart Warm Start (Live Power -> Model Domain)
        3. Calibration (Model Output -> Real World)
        """
        if not self.loaded:
            return {"error": "Models not loaded"}

        # --- 1. Parse Current State from Live Devices ---
        current_state = {
            'lineA_mode': 'OFF', 'lineA_speed_pct': 0,
            'lineB_mode': 'OFF', 'lineB_speed_pct': 0,
            'hvac_mode': 'OFF', 'lighting_mode': 'OFF',
            'prodA_uph': 0, 'prodB_uph': 0
        }

        live_total_power = 0.0
        count_a = 0
        count_b = 0

        if live_devices:
            for d in live_devices:
                # Accumulate actual power for Smart Warm Start
                live_total_power += float(d.get('power', 0))

                name = (d.get('name') or '').lower()
                line_id = (d.get('lineId') or '').lower()
                mode = d.get('mode', 'OFF')
                speed = float(d.get('speed', 0))
                is_running = (mode == 'RUNNING')

                # Production Rate Logic
                if 'line-a' in line_id and d.get('isController'):
                    current_state['prodA_uph'] = d.get('productionRate', 0)
                if 'line-b' in line_id and d.get('isController'):
                    current_state['prodB_uph'] = d.get('productionRate', 0)

                # Custom Name Parsing (Your Logic)
                if 'cnc' in name or 'robotic' in name or 'conveyor' in name:
                    if is_running:
                        current_state['lineA_mode'] = 'RUNNING'
                        current_state['lineA_speed_pct'] += speed
                        count_a += 1
                elif 'painting' in name or 'curing' in name:
                    if is_running:
                        current_state['lineB_mode'] = 'RUNNING'
                        current_state['lineB_speed_pct'] += speed
                        count_b += 1
                elif 'hvac' in name:
                    current_state['hvac_mode'] = 'RUNNING' if is_running else 'OFF'
                elif 'light' in name:
                    current_state['lighting_mode'] = 'RUNNING' if is_running else 'OFF'

            if count_a > 0: current_state['lineA_speed_pct'] /= count_a
            if count_b > 0: current_state['lineB_speed_pct'] /= count_b

        # --- 2. Apply User Overrides (What-If Analysis) ---
        if user_overrides:
            for key, val in user_overrides.items():
                current_state[key] = val

        results = {'total_consumption_kwh': 0, 'total_solar_kwh': 0, 'net_grid_kwh': 0}
        start_time = datetime.now()
        
        # --- 3. SMART WARM START + SCALING ---
        # Logic: Live Power is Real World (Scale B). Model expects Small Numbers (Scale A).
        # We must divide Live Power by Calibration Factor to initialize the model correctly.
        
        if live_total_power > 1000:
            real_world_start_power = live_total_power
            logging.info(f"⚡ Smart Warm Start: Using Live Power {real_world_start_power} W")
        else:
            # Fallback if system is OFF
            is_active = False
            for k, v in current_state.items():
                if 'mode' in k and str(v).upper() in ['RUNNING', 'TURBO', 'NORMAL', '2', '1', 'ON']:
                    is_active = True
            real_world_start_power = 18000 if is_active else 500
            logging.info(f"⚡ Smart Warm Start: Using Static Baseline {real_world_start_power} W")
        
        # CONVERT TO MODEL DOMAIN
        model_domain_start_power = real_world_start_power / CALIBRATION_FACTOR
        
        # Initialize History Lags (In Model Domain)
        current_lag_1 = model_domain_start_power
        current_lag_5 = model_domain_start_power
        self.power_history = [model_domain_start_power] * 15
        
        # --- 4. Prediction Loop ---
        for i in range(minutes_ahead):
            sim_time = start_time + timedelta(minutes=i+1)
            current_rolling_15 = np.mean(self.power_history[-15:])
            
            input_row = {
                'lineA_speed_pct': current_state.get('lineA_speed_pct', 80),
                'lineB_speed_pct': current_state.get('lineB_speed_pct', 80),
                'lineA_mode': self._encode_value('lineA_mode', current_state.get('lineA_mode', 'RUNNING')),
                'lineB_mode': self._encode_value('lineB_mode', current_state.get('lineB_mode', 'RUNNING')),
                'hvac_mode': self._encode_value('hvac_mode', current_state.get('hvac_mode', 'RUNNING')),
                'lighting_mode': self._encode_value('lighting_mode', current_state.get('lighting_mode', 'RUNNING')),
                'prodA_uph': current_state.get('prodA_uph', 3500),
                'prodB_uph': current_state.get('prodB_uph', 3500),
                'hour': sim_time.hour,
                'day_of_week': sim_time.weekday(), 
                'lag_1': current_lag_1,
                'lag_5': current_lag_5,
                'rolling_mean_15': current_rolling_15
            }
            
            # Predict Load (Output is in Model Domain)
            try:
                df_input = pd.DataFrame([input_row])
                raw_pred = self.model_load.predict(df_input)[0]
            except Exception as e:
                raw_pred = current_lag_1 

            # SCALE UP (Model Domain -> Real World)
            calibrated_pred = raw_pred * CALIBRATION_FACTOR
            pred_load_w = max(0, calibrated_pred)

            # Predict Solar
            solar_input = pd.DataFrame([{'hour': sim_time.hour, 'day_of_year': sim_time.timetuple().tm_yday, 'month': sim_time.month}])
            try:
                pred_solar_w = self.model_solar.predict(solar_input)[0]
            except: 
                pred_solar_w = 0
            pred_solar_w = max(0, pred_solar_w)
            
            # Accumulate (W -> kWh for 1 minute tick)
            results['total_consumption_kwh'] += (pred_load_w / 60000)
            results['total_solar_kwh'] += (pred_solar_w / 60000)
            results['net_grid_kwh'] += ((pred_load_w - pred_solar_w) / 60000)
            
            # UPDATE HISTORY (Keep in Model Domain to maintain stability)
            next_lag = raw_pred 
            
            if len(self.power_history) > 4:
                current_lag_5 = self.power_history[-4]
            current_lag_1 = next_lag
            self.power_history.append(next_lag)

        return results

# Initialize Twin
twin = FactoryDigitalTwin(MODEL_DIR)

# ==========================================
# 2. NLP SETUP
# ==========================================
try:
    logging.info(f"Loading NLP model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    logging.info("NLP model loaded.")
except Exception as e:
    logging.error(f"Failed to load SentenceTransformer: {e}")
    model = None

INTENT_EXAMPLES = {
    "predict_future": ["Predict power", "Forecast grid usage", "What happens if I run Line A", "Predict next 30 mins"],
    "check_anomalies": ["Check anomalies", "Scan for faults", "Any errors?", "health check"],
    "device_status": ["Status of Line A", "Is CNC running", "Show device details"],
    "export_csv": ["Export CSV", "Download data", "Save excel"],
    "hybrid_report": ["Start hybrid optimization", "Generate report", "Create energy audit"],
    "pv_vs_grid": ["PV vs Grid", "Solar vs Grid", "Power mix"],
    "recent_consumption": ["Consumption last 5 minutes", "Recent usage", "Short term history"]
}

if model:
    intent_texts = [ex for val in INTENT_EXAMPLES.values() for ex in val]
    intent_embeddings = model.encode(intent_texts, convert_to_numpy=True)
    intent_labels = [key for key, val in INTENT_EXAMPLES.items() for _ in val]

def _norm_text(t: str):
    return t.lower().strip().translate(str.maketrans("", "", string.punctuation))

FALLBACK_KEYWORDS = {
    "predict_future": ["predict", "forecast", "future", "what if"],
    "check_anomalies": ["anomaly", "anomalies", "error", "fault", "check"],
    "device_status": ["status", "running", "state"],
    "export_csv": ["csv", "export", "download"],
    "hybrid_report": ["hybrid", "report", "optimize"],
    "pv_vs_grid": ["pv", "solar", "grid", "mix"],
    "recent_consumption": ["recent", "consumption", "last"]
}

def predict_intent(text: str):
    txt_norm = _norm_text(text)
    if model:
        try:
            q_emb = model.encode([text], convert_to_numpy=True)
            sims = cosine_similarity(q_emb, intent_embeddings)[0]
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= INTENT_THRESHOLD:
                return intent_labels[best_idx], float(sims[best_idx])
        except: pass
    for label, keys in FALLBACK_KEYWORDS.items():
        if any(k in txt_norm for k in keys): return label, 1.0
    return "unknown", 0.0

def normalize_devices(devices):
    out = []
    for d in devices or []:
        out.append({
            "lineId": d.get("lineId") or d.get("id"),
            "lineName": d.get("lineName") or d.get("name"),
            "isController": bool(d.get("isController", False)),
            "basePower": d.get("basePower", 0),
            "mode": d.get("mode", "OFF"),
            "speed": int(d.get("speed", 100)),
            "productionRate": float(d.get("productionRate", 0)),
            "power": float(d.get("power", 0))
        })
    return out

def factory_post(endpoint, payload, timeout=6):
    try:
        url = f"{FACTORY_API_BASE}{endpoint}"
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 3. CHAT ENDPOINT
# ==========================================
@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json() or {}
    message = payload.get("message", "").strip()
    devices = payload.get("devices", [])
    
    intent, score = predict_intent(message)
    logging.info(f"Msg: '{message}' -> Intent: {intent}")
    message_norm = _norm_text(message)

    # --- 1. PREDICT FUTURE (Uses Custom Twin Logic) ---
    if intent == "predict_future" or "predict" in message_norm:
        minutes = 30
        time_match = re.search(r'(\d+)\s*(min|minute|hour|hr)s?', message.lower())
        if time_match:
            val = int(time_match.group(1))
            if 'hour' in time_match.group(2) or 'hr' in time_match.group(2):
                val *= 60
            minutes = val

        # Overrides (Simple keyword extraction)
        overrides = {}
        if 'turbo' in message.lower(): overrides['lineA_mode'] = 'TURBO'
        if 'eco' in message.lower(): overrides['lineA_mode'] = 'ECO'
        if 'off' in message.lower(): overrides['lineA_mode'] = 'OFF'

        # CALL TWIN
        forecast = twin.predict_net_grid_load(minutes, overrides, live_devices=devices)
        
        if "error" in forecast:
            return jsonify({"reply": "Error: AI Models not loaded correctly on server."})

        load = round(forecast['total_consumption_kwh'], 2)
        solar = round(forecast['total_solar_kwh'], 2)
        net = round(forecast['net_grid_kwh'], 2)
        cost = round(net * 0.15, 2)
        
        return jsonify({
            "reply": f"🔮 **Forecast ({minutes} mins)**\n🏭 Demand: {load} kWh\n☀️ Solar: {solar} kWh\n🔌 Grid: {net} kWh (${cost})"
        })

    # --- 2. CHECK ANOMALIES (Call Factory API) ---
    if intent == "check_anomalies":
        controllers = [d for d in normalize_devices(devices) if d.get("isController")]
        anomalies = []
        for c in controllers:
            res = factory_post("/predict", {
                "line_id": c["lineId"], "operating_mode": c["mode"], 
                "speed": c["speed"], "actual_power": c["power"]
            })
            if res.get("is_anomaly"):
                anomalies.append(f"⚠️ **{c['lineName']}**: {res.get('message')}")
        
        reply = "🚨 **Anomalies Detected:**\n" + "\n".join(anomalies) if anomalies else "✅ All systems operational."
        return jsonify({"reply": reply})

    # --- 3. HYBRID REPORT (Trigger React Action) ---
    if intent == "hybrid_report":
        return jsonify({
            "reply": "Initiating Hybrid Optimization. Capturing baseline, optimizing for 30s, then generating report.",
            "actions": [
                {"type": "set_control_mode", "mode": "HYBRID", "duration_sec": 30},
                {"type": "generate_report"}
            ]
        })

    # --- 4. EXPORT CSV (Trigger React Action) ---
    if intent == "export_csv":
        return jsonify({
            "reply": "Preparing CSV download...",
            "actions": [{"type": "export_csv"}]
        })

    # --- 5. PV vs GRID ---
    if intent == "pv_vs_grid":
        pv_payload = {
            "devices": normalize_devices(devices),
            "tickMs": 1000,
            "pv_capacity_kw": payload.get("pv_capacity_kw", 5),
            "battery_soc": payload.get("battery_soc", 0.5)
        }
        res = factory_post("/pv_vs_grid", pv_payload)
        
        pv = res.get("pv_w", 0)
        grid = res.get("grid_draw_w", 0)
        total = res.get("totalPower", 0)
        
        pv_pct = (pv / total * 100) if total > 0 else 0
        grid_pct = (grid / total * 100) if total > 0 else 0
        
        return jsonify({
            "reply": f"⚡ **Energy Mix**\n☀️ Solar: {pv:.0f} W ({pv_pct:.1f}%)\n🔌 Grid: {grid:.0f} W ({grid_pct:.1f}%)"
        })

    # --- 6. RECENT CONSUMPTION ---
    if intent == "recent_consumption":
        cons_payload = {
            "devices": normalize_devices(devices),
            "ticks": 5, "tickMs": 60000,
            "pv_capacity_kw": payload.get("pv_capacity_kw"),
            "battery_soc": payload.get("battery_soc")
        }
        res = factory_post("/consumption", cons_payload, timeout=10)
        avg = res.get("avg", 0)
        return jsonify({"reply": f"📉 **Recent Trend (last 5 mins)**\nAverage Load: {avg:.0f} W"})

    # --- 7. DEVICE STATUS ---
    if intent == "device_status":
        running = [d["lineName"] for d in normalize_devices(devices) if d["mode"] == "RUNNING"]
        if running:
             return jsonify({"reply": f"🟢 **Running Devices:**\n" + "\n".join(running)})
        return jsonify({"reply": "All devices are currently OFF or IDLE."})

    return jsonify({"reply": "I can help with: 'Predict next 30 mins', 'Check anomalies', 'Start hybrid optimization', 'Export CSV'."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010, debug=True)