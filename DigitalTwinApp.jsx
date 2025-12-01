import React, { useEffect, useMemo, useState, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import { Cpu, Lightbulb, Wind, Droplet } from "lucide-react";
import { motion } from "framer-motion";

// ------------------- DigitalTwinApp -------------------
export default function DigitalTwinApp() {
  const TIMESTEPS_TO_KEEP = 40;
  const [devices, setDevices] = useState(() => generateDevices(12));
  const [running, setRunning] = useState(true);
  const [history, setHistory] = useState([]);
  const [tickMs, setTickMs] = useState(1000);
  const [costPerKwh, setCostPerKwh] = useState(10);
  const tickRef = useRef(null);

  const totalPower = useMemo(
    () => devices.reduce((s, d) => s + d.power, 0),
    [devices]
  );

  const avgPower = useMemo(
    () => (devices.length ? Math.round(totalPower / devices.length) : 0),
    [totalPower, devices.length]
  );

  // Simulation interval
  useEffect(() => {
    if (!running) {
      if (tickRef.current) clearInterval(tickRef.current);
      tickRef.current = null;
      return;
    }

    tickRef.current = setInterval(() => {
      setDevices((prev) => prev.map(randomUpdateDevice));
      setHistory((prev) => {
        const t = new Date();
        const newPoint = {
          time: t.toLocaleTimeString(),
          totalPower: prevPowerFromDevices(devices),
        };
        const next = [...prev, newPoint].slice(-TIMESTEPS_TO_KEEP);
        return next;
      });
    }, tickMs);

    return () => {
      if (tickRef.current) clearInterval(tickRef.current);
      tickRef.current = null;
    };
  }, [running, tickMs, devices]);

  // Sync history
  useEffect(() => {
    const t = new Date();
    setHistory((prev) => {
      const next = [...prev, { time: t.toLocaleTimeString(), totalPower }].slice(
        -TIMESTEPS_TO_KEEP
      );
      return next;
    });
  }, [totalPower]);

  // ------------------- Controls -------------------
  function handleToggleRunning() {
    setRunning((r) => !r);
  }

  function handleReset() {
    setDevices(generateDevices(devices.length));
    setHistory([]);
    setRunning(false);
    setTimeout(() => setRunning(true), 200);
  }

  function handleToggleDevice(id) {
    setDevices((prev) =>
      prev.map((d) =>
        d.id === id
          ? { ...d, state: !d.state, power: !d.state ? randomPower(d.basePower) : 0 }
          : d
      )
    );
  }

  function handleExportCSV() {
    let csv = "time,totalPower,deviceCount\n";
    history.forEach((h) => {
      csv += `${h.time},${h.totalPower},${devices.length}\n`;
    });
    csv += "\n#devices snapshot\n";
    csv += "id,name,type,state,basePower,currentPower,x,y\n";
    devices.forEach((d) => {
      csv += `${d.id},${d.name},${d.type},${d.state},${d.basePower},${d.power},${d.x},${d.y}\n`;
    });
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `digital_twin_export_${new Date().toISOString()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function estimatedCostPerHour() {
    const kw = totalPower / 1000;
    return (kw * costPerKwh).toFixed(2);
  }

  // ------------------- Render -------------------
  return (
    <div className="p-6 font-sans text-slate-800">
      <header className="mb-4">
        <h1 className="text-2xl font-bold">Smart Factory — Digital Twin</h1>
        <p className="text-sm text-slate-600 mt-1">
          Live simulation of devices — generate synthetic data, inspect, export, and
          later connect to ML optimization.
        </p>
      </header>

      <div className="grid grid-cols-3 gap-4">
        {/* Floorplan */}
        <section className="col-span-2 bg-white rounded-2xl shadow p-4 relative h-[520px]">
          <h2 className="text-lg font-semibold mb-2">Factory Floorplan</h2>
          <div className="relative bg-slate-50 h-[440px] rounded-lg border border-dashed border-slate-200 overflow-hidden">
            <div className="absolute inset-0 bg-[repeating-linear-gradient(0deg,#e6eef9_0.5px,#e6eef9_0.5px_20px,#ffffff_20px,#ffffff_20px)] opacity-40 pointer-events-none" />
            {devices.map((d) => (
              <motion.button
                key={d.id}
                className="absolute -translate-x-1/2 -translate-y-1/2 flex items-center gap-2 p-2 rounded-lg shadow-md hover:scale-105"
                style={{ left: `${d.x}%`, top: `${d.y}%`, zIndex: 10 }}
                onClick={() => handleToggleDevice(d.id)}
                initial={{ opacity: 0.6 }}
                animate={{ opacity: d.state ? 1 : 0.5 }}
                transition={{ type: "spring", stiffness: 300 }}
                title={`Click to ${d.state ? "turn off" : "turn on"} ${d.name}`}
              >
                <DeviceIcon type={d.type} />
                <div className="text-left text-xs">
                  <div className="font-medium">{d.name}</div>
                  <div className="text-slate-500">{d.power} W</div>
                </div>
              </motion.button>
            ))}
            <div className="absolute left-3 bottom-3 bg-white/90 rounded-md p-2 text-sm shadow">
              <div className="font-medium">Total: {totalPower} W</div>
              <div className="text-slate-500 text-xs">
                Est. ₹/hr: {estimatedCostPerHour()}
              </div>
            </div>
          </div>
        </section>

        {/* Control Panel */}
        <aside className="col-span-1 space-y-4">
          <div className="bg-white rounded-2xl shadow p-4">
            <h3 className="font-semibold">Simulation Controls</h3>
            <div className="mt-3 flex gap-2">
              <button
                onClick={handleToggleRunning}
                className="px-3 py-2 rounded bg-indigo-600 text-white"
              >
                {running ? "Pause" : "Start"}
              </button>
              <button
                onClick={handleReset}
                className="px-3 py-2 rounded border"
              >
                Reset
              </button>
              <button
                onClick={handleExportCSV}
                className="px-3 py-2 rounded border"
              >
                Export CSV
              </button>
            </div>

            <div className="mt-3">
              <label className="text-sm">Tick (ms)</label>
              <input
                type="range"
                min={200}
                max={2000}
                step={100}
                value={tickMs}
                onChange={(e) => setTickMs(Number(e.target.value))}
              />
              <div className="text-xs text-slate-500">{tickMs} ms</div>
            </div>

            <div className="mt-3">
              <label className="text-sm">Cost per kWh</label>
              <div className="flex items-center gap-2 mt-1">
                <input
                  type="number"
                  value={costPerKwh}
                  onChange={(e) => setCostPerKwh(Number(e.target.value))}
                  className="w-24 p-2 border rounded"
                />
                <div className="text-xs text-slate-500">
                  Estimated cost/hr: ₹{estimatedCostPerHour()}
                </div>
              </div>
            </div>
          </div>

          {/* Power Chart */}
          <div className="bg-white rounded-2xl shadow p-4">
            <h3 className="font-semibold">Power Trend (last {TIMESTEPS_TO_KEEP} ticks)</h3>
            <div style={{ width: "100%", height: 220 }} className="mt-2">
              <ResponsiveContainer>
                <LineChart data={history} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" hide />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="totalPower"
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Device list */}
          <div className="bg-white rounded-2xl shadow p-4">
            <h3 className="font-semibold">Devices ({devices.length})</h3>
            <div className="mt-3 max-h-48 overflow-auto">
              {devices.map((d) => (
                <div key={d.id} className="flex items-center justify-between py-2 border-b last:border-b-0">
                  <div className="flex items-center gap-3">
                    <DeviceIcon type={d.type} className="w-5 h-5" />
                    <div>
                      <div className="text-sm font-medium">{d.name}</div>
                      <div className="text-xs text-slate-500">{d.type} • {d.x}% , {d.y}%</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">{d.power} W</div>
                    <div className="text-xs text-slate-500">{d.state ? 'ON' : 'OFF'}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>

      <footer className="mt-6 text-sm text-slate-600">
        Next steps: connect this frontend to a backend (Flask/Node) to gather telemetry and return optimized operating schedules.
      </footer>
    </div>
  );
}

// ------------------- Device Helpers -------------------
function DeviceIcon({ type, className = "w-6 h-6" }) {
  const common = { className };
  if (type === "Machine") return <Cpu {...common} />;
  if (type === "Light") return <Lightbulb {...common} />;
  if (type === "Fan") return <Wind {...common} />;
  if (type === "Cooler") return <Droplet {...common} />;
  return <Cpu {...common} />;
}

function prevPowerFromDevices(devices) {
  if (!devices || devices.length === 0) return 0;
  return devices.reduce((s, d) => s + (d.power ?? 0), 0);
}

function randomPower(base) {
  const factor = 0.6 + Math.random() * 1.2;
  return Math.round(base * factor);
}

function randomUpdateDevice(d) {
  const toggleProb = 0.02;
  let state = d.state;
  if (Math.random() < toggleProb) state = !state;
  const power = state ? randomPower(d.basePower) : 0;
  return { ...d, state, power };
}

function generateDevices(n) {
  const types = [
    { type: "Machine", base: 1400 },
    { type: "Cooler", base: 800 },
    { type: "Fan", base: 150 },
    { type: "Light", base: 60 },
  ];
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = types[Math.floor(Math.random() * types.length)];
    const x = Math.round(Math.random() * 88) + 6;
    const y = Math.round(Math.random() * 88) + 6;
    const state = Math.random() > 0.2;
    const basePower = t.base;
    const power = state ? randomPower(basePower) : 0;
    out.push({
      id: `dev-${i + 1}`,
      name: `${t.type}-${i + 1}`,
      type: t.type,
      basePower,
      power,
      state,
      x,
      y,
    });
  }
  return out;
}
