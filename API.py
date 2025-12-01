import os
import math
import time as _time
from time import time as now
from datetime import datetime, timezone
from collections import defaultdict
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

app = Flask(__name__)
CORS(app)

# ---------------- Config (tune as needed) ----------------
BASE_BY_LINE = {
    "line-a": 3500,
    "line-b": 2800,
    "ancillary-light": 1800,
    "ancillary-hvac": 5000,
}
ONLY_WHEN_RUNNING = (os.getenv("ONLY_WHEN_RUNNING", "1") == "1")
WARMUP_SECONDS = float(os.getenv("ANOMALY_WARMUP_SECONDS", "4"))
REL_MARGIN = float(os.getenv("ANOMALY_REL_MARGIN", "0.10"))
ABS_MARGIN = float(os.getenv("ANOMALY_ABS_MARGIN", "25"))
MAX_REASONABLE_SPEED = float(os.getenv("MAX_REASONABLE_SPEED", "200"))

# Hybrid tuning
DEFAULT_POWER_BUDGET  = int(os.getenv("POWER_BUDGET_W", "9000"))
DEFAULT_TARGET_OUTPUT = float(os.getenv("TARGET_OUTPUT_UPH", "160"))
HYBRID_REDUCE_STEP    = float(os.getenv("HYBRID_REDUCE_STEP", "0.90"))
HYBRID_BOOST_STEP     = float(os.getenv("HYBRID_BOOST_STEP",  "1.05"))
HYBRID_HEADROOM_TO_BOOST = float(os.getenv("HYBRID_HEADROOM_TO_BOOST", "0.85"))
MIN_SPEED_PROD        = int(os.getenv("MIN_SPEED_PROD", "50"))
MIN_SPEED_ANC         = int(os.getenv("MIN_SPEED_ANC", "30"))
MAX_SPEED_CAP         = int(os.getenv("MAX_SPEED_CAP", "120"))

# PV & Battery defaults
DEFAULT_PV_CAP_KW = float(os.getenv("DEFAULT_PV_KW", "5.0"))
DEFAULT_BAT_KWH   = float(os.getenv("DEFAULT_BAT_KWH", "20.0"))
MAX_BAT_CHARGE_W  = float(os.getenv("MAX_BAT_CHARGE_W", "5000"))
MAX_BAT_DISCH_W   = float(os.getenv("MAX_BAT_DISCH_W", "5000"))

# Costs / emissions
COST_PER_KWH = float(os.getenv("COST_PER_KWH", "0.13"))
CO2_PER_KWH = float(os.getenv("CO2_PER_KWH", "0.417"))

# Optional debug env var to print PV internals
PV_DEBUG = (os.getenv("PV_DEBUG", "0") == "1")

_last_running = defaultdict(lambda: None)

# ---------------- Helpers ----------------
def expected_normal_power(base, mode, speed):
    speed = min(max(float(speed), 0.0), MAX_REASONABLE_SPEED)
    if mode == 'IDLE':
        return round(base * 0.05)
    if mode == 'ERROR':
        return round(base * 0.10)
    if mode == 'RUNNING':
        sf = (speed / 100.0) ** 1.5
        base_with_speed = base * sf
        osc = math.sin(_time.time() * 2.0)
        jitter = 3 + ((osc + 1) / 2.0) * 2.0
        return round(base_with_speed + jitter)
    return 0

def anomaly_decision(actual, expected):
    threshold = expected * (1.0 + REL_MARGIN) + ABS_MARGIN
    return actual > threshold, threshold

def is_ancillary(dev: dict) -> bool:
    return bool(dev.get('isController')) and (
        dev.get('lineId','').startswith('ancillary') or dev.get('productionRate',0) <= 0
    )

def prod_uph(dev: dict) -> float:
    if not dev.get('isController'):
        return 0.0
    pr = float(dev.get('productionRate', 0))
    if pr <= 0:
        return 0.0
    sp = float(dev.get('speed', 100))
    return pr * (sp / 100.0)

# ---------------- update_factory_state (simulation + HYBRID + PV/BATT) ----------------
@app.route('/update_factory_state', methods=['POST'])
def update_factory_state():
    data = request.get_json(silent=True) or {}
    devices = data.get('devices')
    tick_ms = int(data.get('tickMs', 1000))
    energy_factors = data.get('energyFactors', {})
    control_mode = (data.get('controlMode') or "NORMAL").upper()
    power_budget = int(data.get('powerBudget') or DEFAULT_POWER_BUDGET)
    target_output = float(data.get('targetOutput') or DEFAULT_TARGET_OUTPUT)
    aggressive = bool(data.get('aggressive', False))

    # PV / battery inputs from frontend (fallback to defaults)
    pv_capacity_kw = float(data.get('pv_capacity_kw') or DEFAULT_PV_CAP_KW)
    battery_capacity_kwh = float(data.get('battery_capacity_kwh') or DEFAULT_BAT_KWH)
    # make sure battery_soc is a float in [0,1]
    battery_soc = float(data.get('battery_soc', 0.5))
    battery_soc = min(max(battery_soc, 0.0), 1.0)

    # client time (if provided) to align PV to user's local time
    client_time_iso = data.get('client_time_iso')
    if client_time_iso:
        try:
            now_dt = datetime.fromisoformat(client_time_iso.replace("Z", "+00:00"))
        except Exception:
            now_dt = datetime.now()
    else:
        # use server local time (not forced UTC) — this makes PV behave as expected for local testing
        now_dt = datetime.now()

    if not devices:
        return {"error": "Invalid input, 'devices' array is required."}, 400

    tick_hours = tick_ms / 1000.0 / 3600.0

    # local helpers
    def get_power(base, mode, speed, factor):
        if mode == 'IDLE':  return round(base * 0.05 * factor)
        if mode == 'ERROR': return round(base * 0.10 * factor)
        if mode == 'RUNNING':
            sf = (speed / 100.0) ** 1.5
            bws = base * sf
            osc = math.sin(_time.time() * 2.0)
            jitter = 3 + ((osc + 1) / 2.0) * 2.0
            return max(0, round(bws * factor + jitter))
        return 0

    def simulate_tick(device):
        mode = device.get('mode', 'OFF')
        units = float(device.get('unitsProduced', 0.0))
        speed = float(device.get('speed', 100))
        if mode == 'RUNNING' and device.get('productionRate', 0) > 0:
            units += device.get('productionRate', 0) * (speed / 100.0) * tick_hours
        device['mode'] = mode
        device['unitsProduced'] = units
        return device

    # 1) simulate devices, auto-HVAC, and compute power (with energy_factors)
    updated = [simulate_tick(d.copy()) for d in devices]

    # Auto HVAC mode decision (same rules you had)
    def get_required_hvac_mode(current_devices):
        high_heat_sources = ['Curing Oven', 'CNC Mill']
        medium_heat_sources = ['Painting Booth', 'Robotic Arm']
        is_high = any(d.get('type') in high_heat_sources and d.get('mode') == 'RUNNING' for d in current_devices)
        is_med  = any(d.get('type') in medium_heat_sources and d.get('mode') == 'RUNNING' for d in current_devices)
        if is_high: return 'RUNNING'
        if is_med:  return 'IDLE'
        return 'OFF'

    required_hvac = get_required_hvac_mode(updated)
    for d in updated:
        if d.get('lineId') == 'ancillary-hvac':
            d['mode'] = required_hvac
            break

    # compute each device power using energy_factors (which simulate anomalies)
    for d in updated:
        base = d.get('basePower', 0)
        mode = d.get('mode', 'OFF')
        speed = d.get('speed', 100)
        factor = float(energy_factors.get(d.get('lineId'), 1.0))
        d['power'] = get_power(base, mode, speed, factor)

    total_power = sum(x.get('power', 0) for x in updated)

    # ---------------- PV generation model (use client local time if provided) ----------------
    # Use local client/server hour for solar curve. Smooth bell between 6..18h with small cloudiness
    hour = now_dt.hour + now_dt.minute / 60.0 + now_dt.second / 3600.0
    pv_kw = 0.0
    if pv_capacity_kw > 0 and 6.0 <= hour <= 18.0:
        # normalized 0..1 with peak at 12: use a smooth sine bell
        pv_norm = math.sin(math.pi * (hour - 6.0) / 12.0)
        pv_norm = max(0.0, pv_norm)  # safety
        derate = 0.85  # panel/system derate
        # gentle cloudiness factor between 0.85..1.0 varying with minutes — not deterministic large swings
        minute_of_day = now_dt.hour * 60 + now_dt.minute
        cloudiness = 0.925 + 0.075 * math.sin((minute_of_day / 1440.0) * 2.0 * math.pi)
        cloudiness = max(0.7, min(1.0, cloudiness))
        pv_kw = pv_capacity_kw * pv_norm * derate * cloudiness
        # clamp to capacity
        pv_kw = min(pv_kw, pv_capacity_kw)
    else:
        pv_kw = 0.0

    pv_w = float(round(pv_kw * 1000.0, 2))

    if PV_DEBUG:
        print(f"[PV DEBUG] now_dt={now_dt.isoformat()} hour={hour:.2f} pv_norm={pv_norm if pv_capacity_kw>0 else 'N/A'} pv_kw={pv_kw:.3f} pv_w={pv_w}kW capacity={pv_capacity_kw}kW cloudiness={locals().get('cloudiness','N/A')}")

    # ---------------- PV-first power flow logic ----------------
    # Priority:
    #   1) PV supplies load
    #   2) Battery discharges to cover any remaining deficit (within discharge rate and SOC)
    #   3) Grid supplies remaining deficit
    #   4) If PV > load, excess PV is used to charge battery (within charge rate and SOC). Any remaining surplus is export (negative grid_draw)
    #
    # All charge/discharge calculations are tick-aware (Wh -> W conversion on the tick).
    grid_draw_w = 0.0

    # If no battery capacity configured, treat battery ops as no-op
    if battery_capacity_kwh <= 0:
        # Simple: grid covers deficit after PV
        grid_draw_w = float(round(max(0.0, total_power - pv_w), 2))
    else:
        # available PV to supply load
        pv_available_w = pv_w
        load_w = float(total_power)

        if pv_available_w >= load_w:
            # PV fully covers load
            grid_draw_w = 0.0
            surplus_w = pv_available_w - load_w

            # charge battery with surplus if possible
            # compute how much Wh we can accept this tick
            max_charge_wh = (1.0 - battery_soc) * battery_capacity_kwh * 1000.0
            # limit by max charging power and tick duration
            possible_charge_wh = min(max_charge_wh, MAX_BAT_CHARGE_W * (tick_ms / 1000.0))
            charge_w = (possible_charge_wh / (tick_ms / 1000.0)) if tick_ms > 0 else 0.0
            # clamp to surplus
            charge_w = min(charge_w, surplus_w)
            # apply charge
            if charge_w > 0:
                battery_soc += (charge_w * (tick_ms / 1000.0)) / (battery_capacity_kwh * 1000.0)
                surplus_w -= charge_w

            # any leftover surplus after charging -> export (negative grid draw)
            grid_draw_w = float(round(-max(0.0, surplus_w), 2))
        else:
            # PV insufficient: deficit = load - PV
            deficit_w = load_w - pv_available_w

            # how much Wh we could discharge this tick given SOC and max discharge rate
            max_discharge_wh_available = battery_soc * battery_capacity_kwh * 1000.0
            possible_discharge_wh = min(max_discharge_wh_available, MAX_BAT_DISCH_W * (tick_ms / 1000.0))
            discharge_w = (possible_discharge_wh / (tick_ms / 1000.0)) if tick_ms > 0 else 0.0
            # discharge only up to the deficit
            discharge_w_to_use = min(discharge_w, deficit_w)

            if discharge_w_to_use > 0:
                # battery supplies part (or all) of the deficit
                battery_soc -= (discharge_w_to_use * (tick_ms / 1000.0)) / (battery_capacity_kwh * 1000.0)
                deficit_w -= discharge_w_to_use

            # remaining deficit is drawn from grid
            grid_draw_w = float(round(deficit_w, 2))

    # clamp SOC
    battery_soc = min(max(battery_soc, 0.0), 1.0)

    # ---------------- If not HYBRID, return quickly ----------------
    if control_mode != "HYBRID":
        return jsonify({
            "devices": updated,
            "totalPower": total_power,
            "pv_w": round(pv_w, 2),
            "grid_draw_w": grid_draw_w,
            "battery_soc": round(battery_soc, 3),
            "optimizationActions": []
        }), 200

    # ---------------- HYBRID SMART MODE ----------------
    actions = []

    def eei(dev):
        p = max(1.0, float(dev.get('power', 0)))
        u = prod_uph(dev)
        return (u * 1000.0 / p) if u > 0 else 0.0

    controllers = [d for d in updated if d.get('isController')]
    running_ctrls = [d for d in controllers if d.get('mode') == 'RUNNING']
    prod_running = [d for d in running_ctrls if not is_ancillary(d)]
    anc_running = [d for d in running_ctrls if is_ancillary(d)]

    # A) Protection: detect extreme overconsumption and IDLE whole line
    overlined_ids = set()
    def expected_normal_power_for_dev(dev):
        base = BASE_BY_LINE.get(dev.get('lineId'), dev.get('basePower', 2500))
        return expected_normal_power(base, dev.get('mode', 'RUNNING'), dev.get('speed', 100))
    def is_extreme_over(actual, expected):
        return actual > (expected * 1.5 + 100)
    for dev in updated:
        if dev.get('isController') and dev.get('mode') == 'RUNNING':
            expected_val = expected_normal_power_for_dev(dev)
            if is_extreme_over(dev.get('power', 0), expected_val):
                overlined_ids.add(dev.get('lineId'))
    if overlined_ids:
        for line_id in overlined_ids:
            line_name = next((d['lineName'] for d in updated if d['lineId'] == line_id and d.get('isController')), line_id)
            actions.append(f"🛡️ Protective action: {line_name} set to IDLE (extreme overconsumption).")
            for dev in updated:
                if dev.get('lineId') == line_id:
                    dev['mode'] = 'IDLE'
                    factor = float(energy_factors.get(line_id, 1.0))
                    dev['power'] = get_power(dev.get('basePower', 0), 'IDLE', dev.get('speed', 100), factor)
            total_power = sum(x.get('power', 0) for x in updated)

    # B) Idle ancillary first when over budget
    if total_power > power_budget and anc_running:
        actions.append(f"HYBRID: Over budget {total_power}W > {power_budget}W → IDLE ancillary first.")
        for d in sorted(anc_running, key=lambda x: x.get('power', 0), reverse=True):
            if total_power <= power_budget: break
            d['mode'] = 'IDLE'
            factor = float(energy_factors.get(d.get('lineId'), 1.0))
            d['power'] = get_power(d.get('basePower', 0), 'IDLE', d.get('speed', 100), factor)
            actions.append(f"↓ {d['lineName']}: RUNNING → IDLE")
            total_power = sum(x.get('power', 0) for x in updated)

    # C) Throttle by EEI (lowest EEI first) if still over budget
    if total_power > power_budget and running_ctrls:
        actions.append("HYBRID: Still over budget → throttling low-EEI controllers.")
        reduce_step = HYBRID_REDUCE_STEP * (0.85 if aggressive else 1.0)
        for d in sorted(running_ctrls, key=lambda x: (eei(x), 0 if is_ancillary(x) else 1)):
            if total_power <= power_budget: break
            old_speed = int(d.get('speed', 100))
            floor = MIN_SPEED_ANC if is_ancillary(d) else MIN_SPEED_PROD
            new_speed = max(floor, int(old_speed * reduce_step))
            if new_speed == old_speed: continue
            d['speed'] = new_speed
            factor = float(energy_factors.get(d.get('lineId'), 1.0))
            d['power'] = get_power(d.get('basePower', 0), 'RUNNING', new_speed, factor)
            actions.append(f"↓ {d['lineName']}: {old_speed}% → {new_speed}% (EEI={eei(d):.2f})")
            total_power = sum(x.get('power', 0) for x in updated)

    # D) If under target output and there's headroom, boost best EEI lines
    prod_total_uph = sum(prod_uph(d) for d in prod_running)
    if prod_total_uph < target_output and total_power < HYBRID_HEADROOM_TO_BOOST * power_budget and prod_running:
        actions.append(f"HYBRID: Output {prod_total_uph:.1f} < target {target_output:.1f} & headroom → boosting efficient lines.")
        boost_step = HYBRID_BOOST_STEP * (1.1 if aggressive else 1.0)
        for d in sorted(prod_running, key=eei, reverse=True):
            if total_power >= power_budget: break
            old_speed = int(d.get('speed', 100))
            new_speed = min(MAX_SPEED_CAP, int(old_speed * boost_step))
            if new_speed == old_speed: continue
            d['speed'] = new_speed
            factor = float(energy_factors.get(d.get('lineId'), 1.0))
            d['power'] = get_power(d.get('basePower', 0), 'RUNNING', new_speed, factor)
            actions.append(f"↑ {d['lineName']}: {old_speed}% → {new_speed}% (EEI={eei(d):.2f})")
            total_power = sum(x.get('power', 0) for x in updated)

    # Final totals after hybrid
    final_total_power = sum(x.get('power', 0) for x in updated)

    return jsonify({
        "devices": updated,
        "totalPower": final_total_power,
        "pv_w": round(pv_w, 2),
        "grid_draw_w": grid_draw_w,
        "battery_soc": round(battery_soc, 3),
        "optimizationActions": actions
    }), 200

# ---------------- ANOMALY DETECTION ENDPOINT ----------------
@app.route('/predict', methods=['POST'])
def predict():
    dp = request.get_json(silent=True) or {}
    if not dp:
        return jsonify({"error": "Invalid input"}), 400
    for k in ['line_id', 'operating_mode', 'speed', 'actual_power']:
        if k not in dp:
            return jsonify({"error": f"Missing key '{k}'"}), 400

    line_id = dp['line_id']
    mode    = dp['operating_mode']
    speed   = float(dp['speed'])
    actual  = float(dp['actual_power'])

    if ONLY_WHEN_RUNNING and mode != 'RUNNING':
        return jsonify({
            "actual_power": round(actual, 2),
            "expected_power": None,
            "difference": None,
            "threshold_used": None,
            "warmup": False,
            "is_anomaly": False,
            "message": "Normal (not RUNNING)"
        }), 200

    ts = now()
    last = _last_running.get(line_id)
    if mode == 'RUNNING' and (last is None or ts - last > 3600):
        _last_running[line_id] = ts
        last = ts
    in_warmup = (mode == 'RUNNING' and last is not None and (ts - last) < WARMUP_SECONDS)

    base = BASE_BY_LINE.get(line_id, 2500.0)
    expected = expected_normal_power(base, mode, speed)
    is_anom, threshold = anomaly_decision(actual, expected)
    if in_warmup:
        is_anom = False

    return jsonify({
        "actual_power": round(actual, 2),
        "expected_power": round(expected, 2),
        "difference": round(actual - expected, 2),
        "threshold_used": round(threshold, 2),
        "warmup": bool(in_warmup),
        "is_anomaly": bool(is_anom),
        "message": "Anomaly Detected!" if is_anom else ("Warming up..." if in_warmup else "Normal")
    }), 200

# ---------------- Reporting helpers & /generate_report ----------------
def parse_history_times_and_powers(history):
    times = []
    powers = []
    for h in history:
        ts = h.get('timestamp')
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            try:
                t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
            except Exception:
                continue
        total = h.get('totalPower')
        if total is None:
            total = sum(d.get('power', 0) for d in (h.get('devices') or []))
        times.append(t)
        powers.append(float(total))
    if len(times) > 1:
        combined = sorted(zip(times, powers), key=lambda x: x[0])
        times, powers = zip(*combined)
        times = list(times); powers = list(powers)
    return list(times), list(powers)

def compute_energy_kwh_from_power_series(times, powers):
    if len(times) < 2:
        avg_watts = float(np.mean(powers)) if len(powers) else 0.0
        return 0.0, [], 0.0
    total_kwh = 0.0
    contributions = []
    for i in range(1, len(times)):
        dt_seconds = (times[i] - times[i-1]).total_seconds()
        if dt_seconds <= 0:
            continue
        avg_w = (powers[i] + powers[i-1]) / 2.0
        kwh = (avg_w * dt_seconds) / 3600.0 / 1000.0
        contributions.append(kwh)
        total_kwh += kwh
    duration_hours = (times[-1] - times[0]).total_seconds() / 3600.0
    return total_kwh, contributions, duration_hours

@app.route('/generate_report', methods=['POST'])
def generate_report():
    payload = request.get_json(silent=True) or {}
    history = payload.get('history') or []
    baseline = payload.get('baselineHistory') or []
    anomaly_hist = payload.get('anomalyHistory') or {}

    times_cur, powers_cur = parse_history_times_and_powers(history)
    times_base, powers_base = parse_history_times_and_powers(baseline)

    kwh_cur, contribs_cur, dur_hours_cur = compute_energy_kwh_from_power_series(times_cur, powers_cur)
    kwh_base, contribs_base, dur_hours_base = compute_energy_kwh_from_power_series(times_base, powers_base)

    avg_kw_cur = (kwh_cur / dur_hours_cur) if dur_hours_cur > 0 else (np.mean(powers_cur)/1000.0 if powers_cur else 0.0)
    avg_kw_base = (kwh_base / dur_hours_base) if dur_hours_base > 0 else (np.mean(powers_base)/1000.0 if powers_base else 0.0)

    norm_kwh_cur_1h = avg_kw_cur * 1.0
    norm_kwh_base_1h = avg_kw_base * 1.0

    cost_cur = kwh_cur * COST_PER_KWH
    cost_base = kwh_base * COST_PER_KWH
    co2_cur = kwh_cur * CO2_PER_KWH
    co2_base = kwh_base * CO2_PER_KWH

    cost_norm_cur = norm_kwh_cur_1h * COST_PER_KWH
    cost_norm_base = norm_kwh_base_1h * COST_PER_KWH
    co2_norm_cur = norm_kwh_cur_1h * CO2_PER_KWH
    co2_norm_base = norm_kwh_base_1h * CO2_PER_KWH

    use_normalized = False
    if dur_hours_base > 0 and dur_hours_cur > 0:
        rel_diff = abs(dur_hours_base - dur_hours_cur) / max(dur_hours_base, dur_hours_cur)
        if rel_diff > 0.05 or dur_hours_base < 0.1 or dur_hours_cur < 0.1:
            use_normalized = True

    if use_normalized and dur_hours_base > 0 and dur_hours_cur > 0:
        savings_kwh = max(0.0, norm_kwh_base_1h - norm_kwh_cur_1h)
        savings_pct = (savings_kwh / norm_kwh_base_1h * 100.0) if norm_kwh_base_1h else 0.0
        savings_cost = (cost_norm_base - cost_norm_cur)
        savings_co2 = max(0.0, co2_norm_base - co2_norm_cur)
        comparison_note = (
            "Comparison normalized to 1-hour equivalents because baseline/current durations differ "
            f"(baseline {dur_hours_base:.2f} h, current {dur_hours_cur:.2f} h)."
        )
    else:
        if kwh_base:
            savings_kwh = max(0.0, (kwh_base - kwh_cur))
            savings_pct = (savings_kwh / kwh_base * 100.0) if kwh_base else 0.0
            savings_cost = cost_base - cost_cur
            savings_co2 = max(0.0, co2_base - co2_cur)
            comparison_note = "Raw totals used for comparison (durations similar or baseline missing)."
        else:
            savings_kwh = 0.0
            savings_pct = 0.0
            savings_cost = 0.0
            savings_co2 = 0.0
            comparison_note = "No baseline provided — cannot compute savings."

    anomaly_counts = {}
    total_checks = 0
    total_anomalies = 0
    if isinstance(anomaly_hist, dict):
        for line, checks in anomaly_hist.items():
            cnt = sum(1 for c in checks if c.get('is_anomaly'))
            total = len(checks)
            anomaly_counts[line] = {"anomalies": cnt, "checks": total}
            total_checks += total
            total_anomalies += cnt
    anomaly_frequency_pct = (total_anomalies / total_checks * 100.0) if total_checks else 0.0

    # Build PDF
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1 summary
        fig1 = plt.figure(figsize=(8.27, 11.69))
        fig1.clf()
        ax = fig1.add_subplot(111)
        ax.axis('off')
        title = "Smart Factory — Energy Audit Report"
        date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        summary_lines = [
            f"{title}",
            f"Generated: {date_str}",
            "",
            "SUMMARY",
            f"Samples (current): {len(times_cur)} samples — duration {dur_hours_cur:.3f} h",
            f"Samples (baseline): {len(times_base)} samples — duration {dur_hours_base:.3f} h" if len(times_base) else "Baseline: not provided",
            "",
            "ENERGY (raw totals):",
            f"Total energy (current): {kwh_cur:.6f} kWh",
            f"Total energy (baseline): {kwh_base:.6f} kWh" if len(times_base) else "",
            f"Estimated cost (current): ${cost_cur:.3f}",
            f"Estimated cost (baseline): ${cost_base:.3f}" if len(times_base) else "",
            f"Estimated CO₂ (current): {co2_cur:.3f} kg",
            f"Estimated CO₂ (baseline): {co2_base:.3f} kg" if len(times_base) else "",
            "",
            "ENERGY (normalized, 1-hour equivalents):",
            f"Normalized (current): {norm_kwh_cur_1h:.6f} kWh / 1h (avg {avg_kw_cur:.3f} kW)",
            f"Normalized (baseline): {norm_kwh_base_1h:.6f} kWh / 1h (avg {avg_kw_base:.3f} kW)" if len(times_base) else "",
            "",
            "SAVINGS",
            f"Energy saved: {savings_kwh:.6f} kWh ({savings_pct:.2f}% )",
            f"Cost saved: ${savings_cost:.3f}",
            f"CO₂ avoided: {savings_co2:.3f} kg",
            "",
            "ANOMALY METRICS",
            f"Total anomaly checks: {total_checks}",
            f"Total anomalies detected: {total_anomalies}",
            f"Anomaly frequency: {anomaly_frequency_pct:.2f}%",
            "",
            f"Comparison note: {comparison_note}",
        ]
        baseline_too_short = len(times_base) > 0 and dur_hours_base < 0.05
        if baseline_too_short:
            summary_lines.append("")
            summary_lines.append("WARNING: Baseline duration is very short — results may not be reliable. Capture baseline for at least 30 seconds.")
        y0 = 0.95
        dy = 0.033
        ax.text(0.02, y0, summary_lines[0], fontsize=16, weight='bold', transform=fig1.transFigure)
        for i, ln in enumerate(summary_lines[1:], start=1):
            ax.text(0.02, y0 - i * dy, ln, fontsize=9, transform=fig1.transFigure)
        pdf.savefig(fig1, bbox_inches='tight')
        plt.close(fig1)

        # Page 2: Time series
        if times_cur and powers_cur:
            fig2 = plt.figure(figsize=(11,6))
            plt.plot(times_cur, powers_cur, label="Current", linewidth=2)
            if times_base and powers_base:
                plt.plot(times_base, powers_base, label="Baseline", linewidth=2, linestyle='--')
            plt.title("Total Power Over Time")
            plt.xlabel("Time")
            plt.ylabel("Power (W)")
            plt.legend()
            plt.grid(True, alpha=0.2)
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            plt.xticks(rotation=25)
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)

        # Page 3: Bar chart
        fig3 = plt.figure(figsize=(8,4))
        labels = []
        values = []
        colors = []
        if len(times_base):
            labels.append("Baseline (raw kWh)")
            values.append(kwh_base)
            colors.append('#a3a0f7')
        labels.append("Current (raw kWh)")
        values.append(kwh_cur)
        colors.append('#6b66e6')
        plt.subplot(1,2,1)
        plt.bar(labels, values, color=colors)
        plt.title("Raw energy totals")
        for i, v in enumerate(values):
            plt.text(i, v*1.01 if v>=0 else v-0.01, f"{v:.6f} kWh", ha='center')
        plt.subplot(1,2,2)
        nlabels = []
        nvals = []
        ncolors = []
        if len(times_base):
            nlabels.append("Baseline (1h)")
            nvals.append(norm_kwh_base_1h)
            ncolors.append('#a3a0f7')
        nlabels.append("Current (1h)")
        nvals.append(norm_kwh_cur_1h)
        ncolors.append('#6b66e6')
        plt.bar(nlabels, nvals, color=ncolors)
        plt.title("Normalized (1-hour equiv.)")
        for i, v in enumerate(nvals):
            plt.text(i, v*1.01 if v>=0 else v-0.01, f"{v:.6f} kWh", ha='center')
        plt.tight_layout()
        pdf.savefig(fig3, bbox_inches='tight')
        plt.close(fig3)

        # Page 4: anomaly table
        if anomaly_counts:
            fig4 = plt.figure(figsize=(8.27, 11.69))
            fig4.clf()
            ax = fig4.add_subplot(111)
            ax.axis('off')
            ax.text(0.02, 0.95, "Anomaly Summary (per line)", fontsize=14, weight='bold', transform=fig4.transFigure)
            y = 0.9
            dy = 0.04
            ax.text(0.02, y, "Line ID     Anomalies     Checks     Frequency", fontsize=10, transform=fig4.transFigure)
            for i, (line, info) in enumerate(anomaly_counts.items(), start=1):
                an = info['anomalies']
                ch = info['checks']
                freq = (an / ch * 100.0) if ch else 0.0
                ax.text(0.02, y - i * dy, f"{line:<12}    {an:<8}    {ch:<8}    {freq:.2f}%", fontsize=10, transform=fig4.transFigure)
            pdf.savefig(fig4, bbox_inches='tight')
            plt.close(fig4)

    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="energy_audit_report_normalized.pdf", mimetype="application/pdf")

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
