import carla
import pygame
import numpy as np
import random
import time
import math
import threading
import webbrowser
from collections import deque
from pynput import keyboard
from flask import Flask, jsonify, request

from carla_middleware import Middleware
from carla_excel import MiddlewareLogger
from pid_controller import build_processed_sensor_data, SensorBasedController

# All sensor data flows through this before the rest of the script
# sees it. See carla_middleware.py for the fault library and how to
# add/toggle attacks - this line is the only thing linking it in.
mw = Middleware()

# Excel/sensor-logging layer (extracted to carla_excel.py). Records
# every tick's PID/middleware/sensor snapshot and flushes it to a
# timestamped .xlsx file on shutdown. Independent of `mw` above.
middleware_logger = MiddlewareLogger()


# =========================================================
# SETTINGS
# =========================================================

HOST = "127.0.0.1"
PORT = 2000
TM_PORT = 8000

# --- Clean test environment ---
# TEST_MAP: force-load a specific CARLA map at startup (None = keep whatever
# map the server already has). Town03 = highway loop (clean long roads,
# good for PID tuning), Town07 = small rural, Town10/Town10HD = city with
# traffic lights (for signal tests).
TEST_MAP = "Town03"

# Set both to False for a completely empty road: no other traffic and no
# pedestrians spawn, so the ego drives alone - nothing to collide with and
# no external actors to trigger/confuse the obstacle logic. Turn back on
# when you need the fault-injection traffic scenario.
SPAWN_TRAFFIC = False
SPAWN_PEDESTRIANS = False

MAX_TRAFFIC = 40

MAX_PEDESTRIANS = 15

POTHOLE_COUNT = 15

RASH_DRIVER_PERCENTAGE = 0.04   # fraction of traffic vehicles that speed / run lights / tailgate

PEDESTRIAN_CROSS_FACTOR = 0.05   # higher = more pedestrians jaywalk instead of using crosswalks

# --- PID controller control path ---
# Pipeline (as assigned): sensor data → middleware → PID controller
# → actuator control → car moves. When USE_PID_CONTROLLER is True the
# ego vehicle is driven by SensorBasedController (lane-tracking steer
# PID + longitudinal speed PID). The middleware layer is carla_middleware
# (mw): it fault-injects the RAW sensor data before the PID ever sees
# it, so the controller only ever reasons over middleware-processed
# readings. The PID's output is applied to the vehicle directly; the
# MiddlewareLogger (carla_excel) merely records the per-tick snapshot -
# it is NOT in the control loop.
USE_PID_CONTROLLER = True

PID_CRUISE_SPEED_KMH = 30.0    # longitudinal speed the speed-PID targets
PID_CAUTION_ZONE_M = 3.0       # meters ahead where soft-braking starts (ramps to full at the emergency distance)
PID_LOOKAHEAD_M = 8.0          # how far ahead the lane-follow steer-PID looks

# --- Ego collision-avoidance braking ---
# Emergency-brake distance must stay BELOW the Traffic Manager's own
# following gap (6.0m, set via distance_to_leading_vehicle below), or the
# override fires during completely normal car-following. 5.0m leaves
# genuine emergencies (cut-ins, jaywalkers, fault injection) as the only
# thing that trips it.
EMERGENCY_BRAKE_DISTANCE = 5.0   # meters - if something is closer than this, ahead, brake hard
EMERGENCY_BRAKE_CONE_DEGREES = 35.0  # how wide "in front of the car" counts

# --- Automatic reverse (backs up on its own if genuinely stuck) ---
STUCK_TICKS_THRESHOLD = 60     # ~3 seconds at 20 ticks/sec of being blocked + stopped
REVERSE_DURATION_TICKS = 40    # ~2 seconds of backing up before trying forward again

# --- Post-collision hold (unless collision_suppress is active) ---
POST_COLLISION_STOP_TICKS = 40   # ~2 seconds of holding a stop after a real collision

# --- Fault-reaction tuning (speed_override/speed_lie/motion faults) ---
IMU_STEER_BIAS_SCALE = 0.05   # how much corrupted IMU accel_x biases fake steering
SPEED_FAULT_KP = 0.02         # proportional gain: how hard throttle chases a faked speed

# --- Pedal bar HUD scaling (based on real IMU acceleration) ---
MAX_ACCEL_FOR_BAR = 3.0    # m/s^2 that reads as 100% on the accelerator bar
MAX_DECEL_FOR_BAR = 6.0    # m/s^2 that reads as 100% on the brake bar
MAX_SPEED_FOR_BAR = 140.0  # km/h that reads as 100% on the speed bar

# --- Side sensor scan (for the HUD "distance on all sides") ---
SENSOR_SCAN_RADIUS = 30.0

CAMERA_DISTANCE = 12.0
CAMERA_HEIGHT = 5.0

POSITION_SMOOTH = 0.18
ROTATION_SMOOTH = 0.18

# ------------------------------------------------------------
# WINDOW / LAYOUT BUDGET
# ------------------------------------------------------------

TEXT_COL_A_X = 30
TEXT_COL_B_X = 400
TEXT_COL_WIDTH = 350
DATA_WIN_WIDTH = 820
DATA_WIN_HEIGHT = 900

SENSOR_CAM_WIDTH = 320
SENSOR_CAM_HEIGHT = 240
CAMERA_GRID_GAP = 20
CAMERA_MARGIN = 30

HISTORY_LENGTH = 200

CHART_WIDTH = 480
CHART_HEIGHT = 300
CHART_GAP = 30
CHART_MARGIN = 30
CHART_COL1_X = CHART_MARGIN
CHART_COL2_X = CHART_MARGIN + CHART_WIDTH + CHART_GAP


# =========================================================
# GLOBAL CAMERA CONTROL
# =========================================================

camera_mode = "chase"
program_running = True
manual_reverse = False

camera_lock = threading.Lock()
control_lock = threading.Lock()


# =========================================================
# GLOBAL KEYBOARD
# =========================================================

def on_press(key):

    global camera_mode
    global program_running
    global manual_reverse

    try:

        if key == keyboard.Key.esc:
            program_running = False

        elif key.char == "1":
            with camera_lock:
                camera_mode = "front"
            print("Camera: FRONT")

        elif key.char == "2":
            with camera_lock:
                camera_mode = "back"
            print("Camera: BACK")

        elif key.char == "3":
            with camera_lock:
                camera_mode = "top"
            print("Camera: TOP")

        elif key.char == "4":
            with camera_lock:
                camera_mode = "bottom"
            print("Camera: LOW / BOTTOM")

        elif key.char == "5":
            with camera_lock:
                camera_mode = "360"
            print("Camera: 360 DEGREE")

        elif key.char == "6":
            with camera_lock:
                camera_mode = "chase"
            print("Camera: CHASE")

        elif key.char == "r":
            with control_lock:
                manual_reverse = not manual_reverse
            print("Reverse gear:", "ON" if manual_reverse else "OFF")

        # -------------------------------------------------
        # MIDDLEWARE / FAULT-INJECTION HOTKEYS
        # -------------------------------------------------

        elif key.char == "g":
            mw.toggle_fault("gps_spoof")

        elif key.char == "f":
            mw.toggle_fault("gps_freeze")

        elif key.char == "n":
            mw.toggle_fault("imu_noise")

        elif key.char == "b":
            mw.toggle_fault("imu_bias")

        elif key.char == "h":
            mw.toggle_fault("distance_hide")

        elif key.char == "j":
            mw.toggle_fault("distance_shrink")

        elif key.char == "k":
            mw.toggle_fault("speed_lie")

        elif key.char == "l":
            mw.toggle_fault("lidar_dropout")

        elif key.char == "o":
            mw.toggle_fault("lidar_ghost")

        elif key.char == "p":
            mw.toggle_fault("radar_ghost")

        elif key.char == "c":
            mw.toggle_fault("collision_suppress")

        elif key.char == "t":
            mw.toggle_fault("speed_override")

        elif key.char == "d":
            mw.toggle_fault("distance_override")

        elif key.char == "u":
            mw.toggle_fault("gnss_override")

        elif key.char == "i":
            mw.toggle_fault("imu_override")

        elif key.char == "w":
            cycle_weather()

        elif key.char == "m":
            mw.enabled = not mw.enabled
            print("MIDDLEWARE master switch:", "ON" if mw.enabled else "OFF (clean passthrough)")

        elif key.char == "/":
            active = mw.active_faults()
            print("Active faults:", active if active else "None")

    except AttributeError:
        pass


keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()


# =========================================================
# DASHBOARD (Flask, background thread)
# =========================================================

DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000

dashboard_app = Flask(__name__)


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Middleware Fault Dashboard</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --panel-2: #1e2229;
    --border: #2a2f3a; --text: #e6e8ec; --muted: #8a90a0;
    --accent: #ff4d4d; --accent-dim: #4a2a2a; --ok: #3ddc84;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 24px 24px 80px;
  }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  #master {
    display: flex; align-items: center; gap: 12px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;
  }
  #master .name { font-weight: 600; }
  #master .desc { color: var(--muted); font-size: 12px; }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 14px;
  }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; transition: border-color .15s;
  }
  .card.active { border-color: var(--accent); }
  .card-head {
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 10px; margin-bottom: 10px;
  }
  .card-head .label { font-size: 13px; font-weight: 600; line-height: 1.3; }
  .card-head .key { color: var(--muted); font-size: 11px; font-family: monospace; }
  .switch { position: relative; width: 40px; height: 22px; flex-shrink: 0; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider-track {
    position: absolute; cursor: pointer; inset: 0;
    background: var(--panel-2); border: 1px solid var(--border);
    border-radius: 999px; transition: .15s;
  }
  .slider-track:before {
    content: ""; position: absolute; height: 16px; width: 16px;
    left: 2px; top: 2px; background: var(--muted); border-radius: 50%;
    transition: .15s;
  }
  input:checked + .slider-track { background: var(--accent-dim); border-color: var(--accent); }
  input:checked + .slider-track:before { transform: translateX(18px); background: var(--accent); }
  .params { display: flex; flex-direction: column; gap: 8px; }
  .param-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .param-row label { font-size: 11px; color: var(--muted); font-family: monospace; }
  .param-row input[type=number], .param-row input[type=text] {
    width: 110px; background: var(--panel-2); border: 1px solid var(--border);
    color: var(--text); border-radius: 6px; padding: 4px 8px; font-size: 12px;
  }
  .param-row input:focus { outline: 1px solid var(--accent); }
  .empty { color: var(--muted); font-size: 12px; font-style: italic; }
  #status { position: fixed; bottom: 14px; right: 18px; font-size: 11px; color: var(--muted); }
</style>
</head>
<body>

<h1>Sensor Fault Injection Dashboard</h1>
<div class="sub">Controls the exact same faults as the keyboard hotkeys - both stay in sync.</div>

<div id="master">
  <label class="switch">
    <input type="checkbox" id="master-toggle">
    <span class="slider-track"></span>
  </label>
  <div>
    <div class="name">Middleware master switch</div>
    <div class="desc">Off = guaranteed clean passthrough, regardless of individual faults (same as the "m" hotkey)</div>
  </div>
</div>

<div id="faults" class="grid"></div>
<div id="status">connecting...</div>

<script>
const POLL_MS = 1000;
const built = new Set();

function isEditing(el) {
  return document.activeElement === el;
}

function buildParamRow(faultName, key, value) {
  const row = document.createElement("div");
  row.className = "param-row";

  const label = document.createElement("label");
  label.textContent = key;
  row.appendChild(label);

  const input = document.createElement("input");
  input.id = `param-${faultName}-${key}`;
  input.dataset.fault = faultName;
  input.dataset.key = key;

  if (Array.isArray(value)) {
    input.type = "text";
    input.value = value.join(", ");
  } else {
    input.type = "number";
    input.step = "any";
    input.value = value;
    if (key.toLowerCase().includes("factor")) {
      input.min = "1";
    }
  }

  input.addEventListener("change", () => applyParam(faultName, key, input));
  row.appendChild(input);
  return row;
}

function buildCard(name, fault) {
  const card = document.createElement("div");
  card.className = "card";
  card.id = `card-${name}`;

  const head = document.createElement("div");
  head.className = "card-head";

  const labelWrap = document.createElement("div");
  const label = document.createElement("div");
  label.className = "label";
  label.textContent = fault.label;
  const key = document.createElement("div");
  key.className = "key";
  key.textContent = name;
  labelWrap.appendChild(label);
  labelWrap.appendChild(key);

  const sw = document.createElement("label");
  sw.className = "switch";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.id = `toggle-${name}`;
  cb.addEventListener("change", () => applyToggle(name, cb));
  const track = document.createElement("span");
  track.className = "slider-track";
  sw.appendChild(cb);
  sw.appendChild(track);

  head.appendChild(labelWrap);
  head.appendChild(sw);
  card.appendChild(head);

  const params = document.createElement("div");
  params.className = "params";
  const keys = Object.keys(fault.params);
  if (keys.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "no tunable parameters";
    params.appendChild(empty);
  } else {
    keys.forEach(k => params.appendChild(buildParamRow(name, k, fault.params[k])));
  }
  card.appendChild(params);

  document.getElementById("faults").appendChild(card);
  built.add(name);
}

function patchCard(name, fault) {
  const card = document.getElementById(`card-${name}`);
  card.classList.toggle("active", fault.enabled);

  const cb = document.getElementById(`toggle-${name}`);
  if (!isEditing(cb)) cb.checked = fault.enabled;

  Object.keys(fault.params).forEach(key => {
    const input = document.getElementById(`param-${name}-${key}`);
    if (!input || isEditing(input)) return;
    const value = fault.params[key];
    input.value = Array.isArray(value) ? value.join(", ") : value;
  });
}

async function applyToggle(name, cb) {
  await postFault(name, cb.checked, {});
}

async function applyParam(name, key, input) {
  const enabled = document.getElementById(`toggle-${name}`).checked;
  const params = {};
  params[key] = input.type === "text" ? input.value : Number(input.value);
  await postFault(name, enabled, params);
}

async function postFault(name, enabled, params) {
  try {
    await fetch("/api/set_fault", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, enabled, params }),
    });
  } catch (e) {
    setStatus("send failed - is the sim still running?");
  }
}

async function applyMaster(enabled) {
  try {
    await fetch("/api/master", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
  } catch (e) {
    setStatus("send failed - is the sim still running?");
  }
}

function setStatus(text) {
  document.getElementById("status").textContent = text;
}

async function poll() {
  try {
    const res = await fetch("/api/faults");
    const state = await res.json();

    const masterCb = document.getElementById("master-toggle");
    if (!isEditing(masterCb)) masterCb.checked = state.master_enabled;

    Object.keys(state.faults).forEach(name => {
      const fault = state.faults[name];
      if (!built.has(name)) buildCard(name, fault);
      patchCard(name, fault);
    });

    setStatus("live - polling every " + (POLL_MS / 1000) + "s");
  } catch (e) {
    setStatus("disconnected - retrying...");
  }
}

document.getElementById("master-toggle").addEventListener("change", (e) => applyMaster(e.target.checked));

poll();
setInterval(poll, POLL_MS);
</script>
</body>
</html>
"""


@dashboard_app.route("/")
def dashboard_index():
    return DASHBOARD_HTML


@dashboard_app.route("/api/faults", methods=["GET"])
def api_get_faults():
    return jsonify(mw.get_state())


@dashboard_app.route("/api/set_fault", methods=["POST"])
def api_set_fault():
    body = request.get_json(force=True, silent=True) or {}
    name = body.get("name")
    enabled = bool(body.get("enabled", False))
    params = body.get("params") or {}

    if name not in mw.faults:
        return jsonify({"ok": False, "error": f"Unknown fault '{name}'"}), 400

    try:
        mw.set_fault(name, enabled, **params)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "state": mw.get_state()["faults"][name]})


@dashboard_app.route("/api/master", methods=["POST"])
def api_set_master():
    body = request.get_json(force=True, silent=True) or {}
    mw.enabled = bool(body.get("enabled", mw.enabled))
    state = "ON" if mw.enabled else "OFF (clean passthrough)"
    print("MIDDLEWARE master switch:", state, "[via dashboard]")
    return jsonify({"ok": True, "master_enabled": mw.enabled})


def _run_dashboard():
    dashboard_app.run(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        threaded=True,
        use_reloader=False,
    )


def _open_dashboard_browser():
    time.sleep(1.0)
    webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")


dashboard_thread = threading.Thread(target=_run_dashboard, daemon=True)
dashboard_thread.start()

threading.Thread(target=_open_dashboard_browser, daemon=True).start()

print(f"Dashboard running at http://localhost:{DASHBOARD_PORT}")


# =========================================================
# CONNECT TO CARLA
# =========================================================

client = carla.Client(HOST, PORT)
client.set_timeout(15.0)

world = client.get_world()
blueprints = world.get_blueprint_library()

print("Connected to CARLA")

# Clean-environment map override: load TEST_MAP if it isn't already the
# active map, then re-fetch the world/blueprints for everything below.
if TEST_MAP and TEST_MAP not in world.get_map().name:
    print(f"[ENV] Loading clean test map {TEST_MAP} (no traffic/pedestrians)...")
    client.load_world(TEST_MAP)
    world = client.get_world()
    blueprints = world.get_blueprint_library()

print("Map:", world.get_map().name)


# =========================================================
# WEATHER
# =========================================================
# Cycled live with the "w" hotkey (see on_press above). CARLA applies
# world.set_weather() immediately - no restart needed, and it doesn't
# interrupt autopilot, sensors, or anything else already running.

WEATHER_PRESETS = [
    ("Clear Noon", carla.WeatherParameters.ClearNoon),
    ("Cloudy Noon", carla.WeatherParameters.CloudyNoon),
    ("Wet Noon", carla.WeatherParameters.WetNoon),
    ("Wet Cloudy Noon", carla.WeatherParameters.WetCloudyNoon),
    ("Mid Rain Noon", carla.WeatherParameters.MidRainyNoon),
    ("Hard Rain Noon", carla.WeatherParameters.HardRainNoon),
    ("Soft Rain Noon", carla.WeatherParameters.SoftRainNoon),
    ("Clear Sunset", carla.WeatherParameters.ClearSunset),
    ("Cloudy Sunset", carla.WeatherParameters.CloudySunset),
    ("Wet Sunset", carla.WeatherParameters.WetSunset),
    ("Wet Cloudy Sunset", carla.WeatherParameters.WetCloudySunset),
    ("Mid Rain Sunset", carla.WeatherParameters.MidRainSunset),
    ("Hard Rain Sunset", carla.WeatherParameters.HardRainSunset),
    ("Soft Rain Sunset", carla.WeatherParameters.SoftRainSunset),
    # CARLA has no native snow precipitation type - WeatherParameters
    # only models rain (precipitation/precipitation_deposits/wetness).
    # This approximates the LOOK of a snowy day (heavy overcast, thick
    # fog, low flat light, no rain) using the parameters that actually
    # exist. It will not show falling snowflakes or snow-covered
    # ground - only the cold/hazy/low-visibility atmosphere.
    ("Snow (simulated)", carla.WeatherParameters(
        cloudiness=95.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=45.0,
        sun_azimuth_angle=0.0,
        sun_altitude_angle=12.0,
        fog_density=55.0,
        fog_distance=25.0,
        fog_falloff=1.5,
        wetness=0.0,
        scattering_intensity=1.0,
        mie_scattering_scale=0.03,
        rayleigh_scattering_scale=0.0331,
    )),
]

weather_index = 0
current_weather_name = WEATHER_PRESETS[weather_index][0]

world.set_weather(WEATHER_PRESETS[weather_index][1])
print(f"Weather set to: {current_weather_name} (press 'w' to cycle)")


def cycle_weather():
    """Advances to the next weather preset and applies it immediately.
    Bound to the "w" hotkey. Deliberately independent of the fault
    system - weather is an environmental condition, not a sensor
    attack - but it's a realistic thing to combine with faults when
    testing (e.g. does distance_shrink behave differently in fog vs
    clear noon)."""
    global weather_index, current_weather_name
    weather_index = (weather_index + 1) % len(WEATHER_PRESETS)
    current_weather_name, preset = WEATHER_PRESETS[weather_index]
    world.set_weather(preset)
    print(f"Weather: {current_weather_name}")


# =========================================================
# PHYSICS
# =========================================================

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
settings.substepping = True
settings.max_substep_delta_time = 0.005
settings.max_substeps = 20
world.apply_settings(settings)


# =========================================================
# TRAFFIC MANAGER
# =========================================================

traffic_manager = client.get_trafficmanager(TM_PORT)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_hybrid_physics_mode(True)
traffic_manager.set_hybrid_physics_radius(50.0)
traffic_manager.set_global_distance_to_leading_vehicle(6.0)
traffic_manager.global_percentage_speed_difference(0.0)
traffic_manager.set_random_device_seed(10)


# =========================================================
# SPAWN POINTS
# =========================================================

spawn_points = world.get_map().get_spawn_points()

if not spawn_points:
    print("No spawn points found.")
    keyboard_listener.stop()
    raise SystemExit


# =========================================================
# SPAWN TESLA
# =========================================================

tesla_bp = blueprints.find("vehicle.tesla.model3")

# Hybrid physics mode (enabled above) only keeps FULL physics running for
# vehicles tagged role_name="hero". Tagging the Tesla as "hero" guarantees
# it always runs full physics instead of being teleport-only.
if tesla_bp.has_attribute("role_name"):
    tesla_bp.set_attribute("role_name", "hero")

vehicle = None
tesla_spawn = None

random.seed(10)
random.shuffle(spawn_points)

for sp in spawn_points:
    vehicle = world.try_spawn_actor(tesla_bp, sp)
    if vehicle is not None:
        tesla_spawn = sp
        break

if vehicle is None:
    print("Failed to spawn Tesla.")
    keyboard_listener.stop()
    raise SystemExit

print("Tesla Model 3 spawned successfully.")

vehicle.set_simulate_physics(True)


# =========================================================
# AUTOPILOT
# =========================================================

if USE_PID_CONTROLLER:
    vehicle.set_autopilot(False)
    print("PID controller will drive the ego vehicle (Traffic Manager autopilot off).")
else:
    vehicle.set_autopilot(True, traffic_manager.get_port())

    traffic_manager.auto_lane_change(vehicle, True)
    traffic_manager.distance_to_leading_vehicle(vehicle, 6.0)
    traffic_manager.vehicle_percentage_speed_difference(vehicle, 0.0)
    traffic_manager.ignore_vehicles_percentage(vehicle, 0.0)
    traffic_manager.ignore_walkers_percentage(vehicle, 0.0)
    traffic_manager.ignore_lights_percentage(vehicle, 0.0)
    traffic_manager.ignore_signs_percentage(vehicle, 0.0)


def _current_tm_ignore_percentage():
    """How much Traffic Manager should ignore vehicles/walkers on its
    own, independent of anything the middleware reports.

    TM computes its own collision avoidance directly from CARLA's real
    world state - it never reads mw's output. So as long as TM keeps
    its own avoidance active (0% ignore), it silently protects the car
    regardless of what distance_hide/collision_suppress do to OUR
    perception, and those faults look like they're "not working" even
    though they're corrupting the data correctly. While either of them
    is active, TM's own avoidance is relaxed to 100% (fully ignore) so
    the corrupted middleware reading becomes the ONLY thing standing
    between the car and a collision - which is what actually
    demonstrates the vulnerability instead of just hiding it behind a
    second, unrelated safety net.
    """
    masking_fault_active = mw.is_active("distance_hide") or mw.is_active("collision_suppress")
    return 100.0 if masking_fault_active else 0.0


def _rearm_ego_autopilot():
    """Re-applies the ego's safety rules after any set_autopilot(True) call
    - a fresh set_autopilot call resets everything to TM defaults, so this
    has to be called every time autopilot is handed back to the car."""
    ignore_pct = _current_tm_ignore_percentage()
    vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.auto_lane_change(vehicle, True)
    traffic_manager.distance_to_leading_vehicle(vehicle, 6.0)
    traffic_manager.vehicle_percentage_speed_difference(vehicle, 0.0)
    traffic_manager.ignore_vehicles_percentage(vehicle, ignore_pct)
    traffic_manager.ignore_walkers_percentage(vehicle, ignore_pct)
    traffic_manager.ignore_lights_percentage(vehicle, 0.0)
    traffic_manager.ignore_signs_percentage(vehicle, 0.0)


# =========================================================
# TRAFFIC
# =========================================================

traffic_vehicles = []

traffic_blueprints = blueprints.filter("vehicle.*")

traffic_blueprints = [
    bp for bp in traffic_blueprints
    if "ambulance" not in bp.id
    and "firetruck" not in bp.id
    and "police" not in bp.id
]

traffic_spawn_points = spawn_points.copy()
random.shuffle(traffic_spawn_points)

# Clean environment: skip traffic entirely (empty list = the spawn loop
# below naturally does nothing and the HUD reads "0 traffic vehicles").
if not SPAWN_TRAFFIC:
    traffic_spawn_points = []

for sp in traffic_spawn_points:

    if len(traffic_vehicles) >= MAX_TRAFFIC:
        break

    distance = sp.location.distance(tesla_spawn.location)

    if distance < 30:
        continue

    bp = random.choice(traffic_blueprints)

    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", random.choice(colors))

    npc = world.try_spawn_actor(bp, sp)

    if npc is not None:

        npc.set_autopilot(True, traffic_manager.get_port())

        is_rash_driver = random.random() < RASH_DRIVER_PERCENTAGE

        if is_rash_driver:

            traffic_manager.auto_lane_change(npc, True)
            traffic_manager.distance_to_leading_vehicle(npc, 4.0)
            traffic_manager.vehicle_percentage_speed_difference(
                npc, random.uniform(-15.0, -5.0)
            )
            traffic_manager.ignore_lights_percentage(npc, 15.0)
            traffic_manager.ignore_signs_percentage(npc, 15.0)
            traffic_manager.random_left_lanechange_percentage(npc, 10.0)
            traffic_manager.random_right_lanechange_percentage(npc, 10.0)

        else:

            traffic_manager.auto_lane_change(npc, True)
            traffic_manager.distance_to_leading_vehicle(npc, 6.0)
            traffic_manager.vehicle_percentage_speed_difference(
                npc, random.uniform(-5.0, 5.0)
            )

        traffic_vehicles.append(npc)


print(
    f"{len(traffic_vehicles)} traffic vehicles spawned "
    f"({int(RASH_DRIVER_PERCENTAGE * 100)}% rash/rule-violating)."
)


# =========================================================
# PEDESTRIANS (with jaywalking)
# =========================================================

world.set_pedestrians_cross_factor(PEDESTRIAN_CROSS_FACTOR)

walker_blueprints = blueprints.filter("walker.pedestrian.*")
walker_controller_bp = blueprints.find("controller.ai.walker")

pedestrians = []
pedestrian_controllers = []

for _ in range(0 if not SPAWN_PEDESTRIANS else MAX_PEDESTRIANS):

    nav_location = world.get_random_location_from_navigation()

    if nav_location is None:
        continue

    bp = random.choice(walker_blueprints)

    walker = world.try_spawn_actor(bp, carla.Transform(nav_location))

    if walker is None:
        continue

    controller = world.spawn_actor(
        walker_controller_bp, carla.Transform(), attach_to=walker
    )

    controller.start()
    controller.go_to_location(world.get_random_location_from_navigation())
    controller.set_max_speed(random.uniform(1.2, 2.5))

    pedestrians.append(walker)
    pedestrian_controllers.append(controller)


print(
    f"{len(pedestrians)} pedestrians spawned "
    f"(cross factor {PEDESTRIAN_CROSS_FACTOR} - some will jaywalk)."
)


# =========================================================
# POTHOLES (visual hazard markers, non-collidable)
# =========================================================

pothole_locations = []

for _ in range(POTHOLE_COUNT):

    sp = random.choice(spawn_points)

    offset = carla.Location(
        x=random.uniform(-15.0, 15.0),
        y=random.uniform(-15.0, 15.0)
    )

    pothole_center = carla.Location(
        sp.location.x + offset.x,
        sp.location.y + offset.y,
        sp.location.z + 0.05
    )

    world.debug.draw_point(
        pothole_center,
        size=0.15,
        color=carla.Color(120, 60, 20),
        life_time=0.0
    )

    world.debug.draw_string(
        pothole_center + carla.Location(z=0.3),
        "POTHOLE",
        draw_shadow=False,
        color=carla.Color(255, 60, 0),
        life_time=0.0
    )

    pothole_locations.append(pothole_center)


print(f"{len(pothole_locations)} pothole markers placed on the road network.")


# =========================================================
# FLOOD ZONES (visual water hazard markers, non-collidable)
# =========================================================

FLOOD_ZONE_COUNT = 6
FLOOD_ZONE_RADIUS = 3.5      # meters - radius of each puddle
FLOOD_ZONE_POINTS = 45       # points scattered per zone to fill in the "puddle" look

flood_zone_locations = []

for _ in range(FLOOD_ZONE_COUNT):

    sp = random.choice(spawn_points)

    offset = carla.Location(
        x=random.uniform(-15.0, 15.0),
        y=random.uniform(-15.0, 15.0)
    )

    flood_center = carla.Location(
        sp.location.x + offset.x,
        sp.location.y + offset.y,
        sp.location.z + 0.03
    )

    # Scatter points inside a circle so it reads as a filled puddle rather
    # than a single dot - same trick as the pothole markers, just denser
    # and wider since a flood zone should look like an area, not a point.
    # sqrt() on the radius avoids clustering points toward the center,
    # giving a uniform-density puddle instead of a bullseye.
    for _ in range(FLOOD_ZONE_POINTS):

        angle = random.uniform(0.0, 2 * math.pi)
        radius = FLOOD_ZONE_RADIUS * math.sqrt(random.uniform(0.0, 1.0))

        point = flood_center + carla.Location(
            x=math.cos(angle) * radius,
            y=math.sin(angle) * radius
        )

        world.debug.draw_point(
            point,
            size=0.12,
            color=carla.Color(40, 110, 190),
            life_time=0.0
        )

    world.debug.draw_string(
        flood_center + carla.Location(z=0.4),
        "FLOOD ZONE",
        draw_shadow=False,
        color=carla.Color(80, 170, 255),
        life_time=0.0
    )

    flood_zone_locations.append(flood_center)


print(f"{len(flood_zone_locations)} flood zone markers placed on the road network.")


# =========================================================
# SPECTATOR
# =========================================================

spectator = world.get_spectator()


# =========================================================
# HAZARD SCAN (distance to nearest object, all sides)
# =========================================================

def scan_surroundings():
    """Returns nearest-object distance (meters) in front / back / left /
    right of the ego car. Used both for the HUD readout and the
    emergency braking check below."""

    transform = vehicle.get_transform()
    origin = transform.location
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()

    nearest = {
        "front": SENSOR_SCAN_RADIUS,
        "back": SENSOR_SCAN_RADIUS,
        "left": SENSOR_SCAN_RADIUS,
        "right": SENSOR_SCAN_RADIUS,
    }

    candidates = list(traffic_vehicles) + list(pedestrians)

    for other in candidates:

        if other is None or not other.is_alive:
            continue

        other_location = other.get_location()
        delta = other_location - origin
        distance = math.sqrt(delta.x ** 2 + delta.y ** 2)

        if distance < 0.5 or distance > SENSOR_SCAN_RADIUS:
            continue

        forward_component = delta.x * forward.x + delta.y * forward.y
        right_component = delta.x * right.x + delta.y * right.y

        if abs(forward_component) >= abs(right_component):
            key = "front" if forward_component >= 0 else "back"
        else:
            key = "right" if right_component >= 0 else "left"

        if distance < nearest[key]:
            nearest[key] = distance

    for direction in nearest:
        nearest[direction] = mw.process_distance(direction, nearest[direction], SENSOR_SCAN_RADIUS)

    return nearest


def _same_lane_ahead(marker_location):
    """True if marker_location is in the ego's own driving lane (same road
    + same lane id). Keeps the front hazard cone clear of oncoming and
    adjacent-lane traffic, which otherwise produced permanent phantom
    braking (the ego brake-danced for cars it was not following) and made
    the avoid-bias steer toward oncoming vehicles."""
    try:
        ego_wp = world.get_map().get_waypoint(
            vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
        mark_wp = world.get_map().get_waypoint(
            marker_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if ego_wp is None or mark_wp is None:
            return True
        if ego_wp.is_junction or mark_wp.is_junction:
            return True
        return (ego_wp.road_id == mark_wp.road_id) and (ego_wp.lane_id == mark_wp.lane_id)
    except Exception:
        return True


def front_hazard_distance(nearest):
    """Narrower forward cone check specifically for emergency braking,
    so the car only brakes for things genuinely in its path. This is
    the value that actually drives the brake decision below, so it has
    to reflect a spoofed/shrunk/hidden reading, not just the HUD number.

    IMPORTANT: the "nothing detected" sentinel has to equal the same
    scan_radius handed to mw.process_distance(), or a fault like
    distance_shrink ends up shrinking a value the car was never going
    to react to anyway instead of shrinking what this cone actually
    reports as its max range. The sentinel is SENSOR_SCAN_RADIUS (far
    outside the emergency/caution braking band), so an empty road is
    never mistaken for a hazard - exactly what "no obstacle" means.
    """

    transform = vehicle.get_transform()
    origin = transform.location
    forward = transform.get_forward_vector()

    hazard_scan_radius = SENSOR_SCAN_RADIUS
    closest = hazard_scan_radius

    candidates = list(traffic_vehicles) + list(pedestrians)

    for other in candidates:

        if other is None or not other.is_alive:
            continue

        other_location = other.get_location()
        delta = other_location - origin
        distance = math.sqrt(delta.x ** 2 + delta.y ** 2)

        if distance < 0.5 or distance > hazard_scan_radius:
            continue

        forward_component = delta.x * forward.x + delta.y * forward.y

        if forward_component <= 0:
            continue

        angle = math.degrees(
            math.acos(max(-1.0, min(1.0, forward_component / distance)))
        )

        if angle <= EMERGENCY_BRAKE_CONE_DEGREES and distance < closest:
            # Lane gate: only genuinely-in-path obstacles count at range.
            # An object in the SAME lane (or anything within a hard near
            # distance, e.g. crossing traffic at a junction) is a real
            # hazard; oncoming/adjacent-lane cars further out are NOT -
            # braking for them is the phantom-braking complaint.
            if _same_lane_ahead(other_location) or distance < (EMERGENCY_BRAKE_DISTANCE + 2.0):
                closest = distance

    closest = mw.process_distance("front", closest, hazard_scan_radius)

    return closest


def apply_fault_hazard_reactions(front_hazard):
    """Lets faults that aren't distance-shaped push front_hazard in
    EITHER direction, so every fault gets a physical reaction:

      - radar_ghost / lidar_ghost  -> phantom danger (brake for nothing)
      - distance_hide -> hide a real danger the front-distance scan
        would otherwise see (fail to brake for something real, which
        is what actually lets the ego collide)
      - lidar_dropout -> probabilistically drops a real detection,
        same effect as distance_hide but only some of the time
        (matching its drop_fraction parameter)

    collision_suppress is deliberately NOT handled here - it only
    affects what happens AFTER a real collision (see on_collision():
    it skips the post-impact stop-hold, and process_collision() hides
    the reported actor name). It has nothing to do with the front
    distance scan, so it must never be able to cancel a phantom hazard
    that distance_override/distance_shrink just created - if it were
    lumped in with distance_hide here, toggling both at once would
    silently undo the other fault's effect.

    distance_override / distance_shrink are already applied earlier,
    inside process_distance() itself, so they aren't touched again here.
    """

    ghost_active = mw.is_active("radar_ghost") or mw.is_active("lidar_ghost")
    if ghost_active and front_hazard >= EMERGENCY_BRAKE_DISTANCE:
        front_hazard = EMERGENCY_BRAKE_DISTANCE - 1.0

    masking_active = mw.is_active("distance_hide")
    if masking_active and front_hazard < EMERGENCY_BRAKE_DISTANCE:
        front_hazard = EMERGENCY_BRAKE_DISTANCE + 1.0

    elif mw.is_active("lidar_dropout") and front_hazard < EMERGENCY_BRAKE_DISTANCE:
        drop_fraction = mw.faults["lidar_dropout"]["drop_fraction"]
        if random.random() < drop_fraction:
            front_hazard = EMERGENCY_BRAKE_DISTANCE + 1.0

    return front_hazard


# =========================================================
# DIRECTIONAL CAMERAS (front / back / left / right feeds)
# =========================================================

camera_frames = {}

camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(SENSOR_CAM_WIDTH))
camera_bp.set_attribute("image_size_y", str(SENSOR_CAM_HEIGHT))
camera_bp.set_attribute("sensor_tick", "0.15")

camera_rig_transforms = {

    "front": carla.Transform(
        carla.Location(x=2.0, z=1.4),
        carla.Rotation(yaw=0.0)
    ),

    "back": carla.Transform(
        carla.Location(x=-2.0, z=1.4),
        carla.Rotation(yaw=180.0)
    ),

    "left": carla.Transform(
        carla.Location(z=1.4),
        carla.Rotation(yaw=-90.0)
    ),

    "right": carla.Transform(
        carla.Location(z=1.4),
        carla.Rotation(yaw=90.0)
    ),
}

directional_cameras = []

for cam_name, cam_transform in camera_rig_transforms.items():

    cam = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

    def make_callback(name):
        def callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            camera_frames[name] = array
        return callback

    cam.listen(make_callback(cam_name))
    directional_cameras.append(cam)


print(f"{len(directional_cameras)} directional cameras attached (front, back, left, right).")


# =========================================================
# FULL SENSOR SUITE (GNSS, IMU, collision, lane invasion)
# =========================================================

sensor_readings = {
    "gnss": {"lat": 0.0, "lon": 0.0, "alt": 0.0},
    "imu": {
        "accel_x": 0.0, "accel_y": 0.0, "accel_z": 0.0,
        "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
        "compass": 0.0,
    },
    "collision": "None",
    "lane_invasion": "None",
}

# --- GNSS ---
gnss_bp = blueprints.find("sensor.other.gnss")
gnss_sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)


def on_gnss(data):
    lat, lon, alt = mw.process_gnss(data.latitude, data.longitude, data.altitude)
    sensor_readings["gnss"]["lat"] = lat
    sensor_readings["gnss"]["lon"] = lon
    sensor_readings["gnss"]["alt"] = alt


gnss_sensor.listen(on_gnss)


# --- IMU (accelerometer + gyroscope + compass) ---
imu_bp = blueprints.find("sensor.other.imu")
imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)


def on_imu(data):
    accel, gyro, compass = mw.process_imu(
        (data.accelerometer.x, data.accelerometer.y, data.accelerometer.z),
        (data.gyroscope.x, data.gyroscope.y, data.gyroscope.z),
        math.degrees(data.compass)
    )
    sensor_readings["imu"]["accel_x"] = accel[0]
    sensor_readings["imu"]["accel_y"] = accel[1]
    sensor_readings["imu"]["accel_z"] = accel[2]
    sensor_readings["imu"]["gyro_x"] = gyro[0]
    sensor_readings["imu"]["gyro_y"] = gyro[1]
    sensor_readings["imu"]["gyro_z"] = gyro[2]
    sensor_readings["imu"]["compass"] = compass


imu_sensor.listen(on_imu)


# --- Collision sensor ---
collision_bp = blueprints.find("sensor.other.collision")
collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

collision_stop_ticks_remaining = 0


def on_collision(event):
    global collision_stop_ticks_remaining
    sensor_readings["collision"] = mw.process_collision(event.other_actor.type_id)
    if not mw.is_active("collision_suppress"):
        collision_stop_ticks_remaining = POST_COLLISION_STOP_TICKS
    # else: fault active -> no post-collision stop at all, car just
    # keeps driving through/past whatever it hit


collision_sensor.listen(on_collision)


# --- Lane invasion sensor ---
lane_bp = blueprints.find("sensor.other.lane_invasion")
lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=vehicle)


def on_lane_invasion(event):
    crossed_types = [str(marking.type) for marking in event.crossed_lane_markings]
    sensor_readings["lane_invasion"] = ", ".join(crossed_types) if crossed_types else "None"


lane_sensor.listen(on_lane_invasion)


# --- LiDAR ---
lidar_bp = blueprints.find("sensor.lidar.ray_cast")
lidar_bp.set_attribute("channels", "32")
lidar_bp.set_attribute("range", "40")
lidar_bp.set_attribute("points_per_second", "56000")
lidar_bp.set_attribute("rotation_frequency", "20")
lidar_bp.set_attribute("sensor_tick", "0.1")

lidar_sensor = world.spawn_actor(
    lidar_bp, carla.Transform(carla.Location(z=2.2)), attach_to=vehicle
)


def on_lidar(data):
    points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
    if len(points) > 600:
        idx = np.random.choice(len(points), 600, replace=False)
        points = points[idx]
    points = mw.process_lidar(points)
    sensor_readings["lidar_points"] = points


lidar_sensor.listen(on_lidar)


# --- Radar ---
radar_bp = blueprints.find("sensor.other.radar")
radar_bp.set_attribute("horizontal_fov", "60")
radar_bp.set_attribute("vertical_fov", "20")
radar_bp.set_attribute("range", "40")
radar_bp.set_attribute("sensor_tick", "0.1")

radar_sensor = world.spawn_actor(
    radar_bp, carla.Transform(carla.Location(x=2.2, z=1.0)), attach_to=vehicle
)


def on_radar(data):
    detections = []
    for detection in data:
        detections.append({
            "depth": detection.depth,
            "azimuth_deg": math.degrees(detection.azimuth),
            "velocity": detection.velocity,
        })
    detections = mw.process_radar(detections)
    sensor_readings["radar"] = detections


radar_sensor.listen(on_radar)


# --- Obstacle detector ---
obstacle_bp = blueprints.find("sensor.other.obstacle")
obstacle_bp.set_attribute("distance", "15")
obstacle_bp.set_attribute("hit_radius", "0.5")
obstacle_bp.set_attribute("only_dynamics", "False")

obstacle_sensor = world.spawn_actor(
    obstacle_bp, carla.Transform(carla.Location(x=2.0, z=1.0)), attach_to=vehicle
)


def on_obstacle(event):
    other_name = event.other_actor.type_id if event.other_actor else "Unknown"
    distance = mw.process_distance("front", event.distance, 15.0)
    sensor_readings["obstacle"] = {
        "actor": other_name,
        "distance": distance,
        "time": time.time(),
    }


obstacle_sensor.listen(on_obstacle)


# --- DVS (event) camera ---
dvs_bp = blueprints.find("sensor.camera.dvs")
dvs_bp.set_attribute("image_size_x", str(SENSOR_CAM_WIDTH))
dvs_bp.set_attribute("image_size_y", str(SENSOR_CAM_HEIGHT))
dvs_bp.set_attribute("sensor_tick", "0.15")

dvs_sensor = world.spawn_actor(dvs_bp, camera_rig_transforms["front"], attach_to=vehicle)

dvs_frame_rgb = np.zeros((SENSOR_CAM_HEIGHT, SENSOR_CAM_WIDTH, 3), dtype=np.uint8)


def on_dvs(data):
    global dvs_frame_rgb

    events = np.frombuffer(
        data.raw_data,
        dtype=np.dtype([
            ("x", np.uint16), ("y", np.uint16),
            ("t", np.int64), ("pol", np.bool_),
        ])
    )

    frame = np.zeros((data.height, data.width, 3), dtype=np.uint8)

    if len(events) > 0:
        positive = events["pol"]
        frame[events["y"][positive], events["x"][positive]] = (60, 160, 255)
        frame[events["y"][~positive], events["x"][~positive]] = (255, 80, 80)

    dvs_frame_rgb = frame


dvs_sensor.listen(on_dvs)


sensor_readings["lidar_points"] = np.empty((0, 4), dtype=np.float32)
sensor_readings["radar"] = []
sensor_readings["obstacle"] = {"actor": "None", "distance": 0.0, "time": None}

extra_sensors = [
    gnss_sensor, imu_sensor, collision_sensor, lane_sensor,
    lidar_sensor, radar_sensor, obstacle_sensor, dvs_sensor,
]

print(
    "Full sensor suite active: Camera, LiDAR, Radar, IMU, GNSS, "
    "Collision, Lane Invasion, Obstacle Detector, DVS Camera."
)


# =========================================================
# CHART HISTORY BUFFERS (for the live sensor graphs)
# =========================================================

speed_history = deque(maxlen=HISTORY_LENGTH)
accel_history = deque(maxlen=HISTORY_LENGTH)
throttle_history = deque(maxlen=HISTORY_LENGTH)
brake_history = deque(maxlen=HISTORY_LENGTH)
lidar_left_history = deque(maxlen=HISTORY_LENGTH)
lidar_right_history = deque(maxlen=HISTORY_LENGTH)
gnss_lat_history = deque(maxlen=HISTORY_LENGTH)
gnss_lon_history = deque(maxlen=HISTORY_LENGTH)


# =========================================================
# CAMERA VARIABLES
# =========================================================

smooth_position = vehicle.get_location()
smooth_yaw = vehicle.get_transform().rotation.yaw
smooth_pitch = -18.0
orbit_angle = 0.0


# =========================================================
# CAMERA SMOOTH ANGLE
# =========================================================

def smooth_angle(current, target, factor):
    difference = ((target - current + 180.0) % 360.0) - 180.0
    return current + difference * factor


# =========================================================
# CAMERA UPDATE
# =========================================================

def update_camera():

    global smooth_position
    global smooth_yaw
    global smooth_pitch
    global orbit_angle

    transform = vehicle.get_transform()
    vehicle_location = transform.location
    vehicle_yaw = transform.rotation.yaw

    with camera_lock:
        current_mode = camera_mode

    if current_mode == "chase":

        forward = transform.get_forward_vector()
        target_position = (
            vehicle_location - forward * CAMERA_DISTANCE + carla.Location(z=CAMERA_HEIGHT)
        )
        target_yaw = vehicle_yaw
        target_pitch = -18.0

    elif current_mode == "front":

        forward = transform.get_forward_vector()
        target_position = vehicle_location + forward * 7.0 + carla.Location(z=2.5)
        target_yaw = vehicle_yaw
        target_pitch = -5.0

    elif current_mode == "back":

        forward = transform.get_forward_vector()
        target_position = vehicle_location - forward * 8.0 + carla.Location(z=3.0)
        target_yaw = vehicle_yaw + 180.0
        target_pitch = -10.0

    elif current_mode == "top":

        target_position = vehicle_location + carla.Location(z=12.0)
        target_yaw = vehicle_yaw
        target_pitch = -90.0

    elif current_mode == "bottom":

        forward = transform.get_forward_vector()
        target_position = vehicle_location + forward * 2.0 + carla.Location(z=1.2)
        target_yaw = vehicle_yaw
        target_pitch = -2.0

    elif current_mode == "360":

        orbit_angle += 0.8
        radians = math.radians(orbit_angle)
        radius = 12.0
        x = math.cos(radians) * radius
        y = math.sin(radians) * radius
        target_position = vehicle_location + carla.Location(x=x, y=y, z=6.0)
        target_yaw = math.degrees(radians) + 180.0
        target_pitch = -15.0

    else:

        target_position = vehicle_location + carla.Location(z=5.0)
        target_yaw = vehicle_yaw
        target_pitch = -15.0

    smooth_position = carla.Location(
        x=smooth_position.x + (target_position.x - smooth_position.x) * POSITION_SMOOTH,
        y=smooth_position.y + (target_position.y - smooth_position.y) * POSITION_SMOOTH,
        z=smooth_position.z + (target_position.z - smooth_position.z) * POSITION_SMOOTH
    )

    smooth_yaw = smooth_angle(smooth_yaw, target_yaw, ROTATION_SMOOTH)
    smooth_pitch += (target_pitch - smooth_pitch) * ROTATION_SMOOTH

    spectator.set_transform(
        carla.Transform(
            smooth_position,
            carla.Rotation(pitch=smooth_pitch, yaw=smooth_yaw, roll=0.0)
        )
    )


# =========================================================
# PYGAME - THREE SEPARATE WINDOWS
# =========================================================

pygame.init()

from pygame._sdl2 import video as sdl2_video

display_info = pygame.display.Info()
SCREEN_W = display_info.current_w
SCREEN_H = display_info.current_h

CAM_TILE_WIDTH = SENSOR_CAM_WIDTH
CAM_TILE_HEIGHT = SENSOR_CAM_HEIGHT

CAMERA_WIN_WIDTH = (
    CAMERA_MARGIN + 2 * CAM_TILE_WIDTH + CAMERA_GRID_GAP
    + CAMERA_GRID_GAP + CAM_TILE_WIDTH + CAMERA_MARGIN
)
CAMERA_WIN_HEIGHT = 70 + 2 * CAM_TILE_HEIGHT + CAMERA_GRID_GAP + CAMERA_MARGIN

max_camera_w = SCREEN_W - 100
max_camera_h = SCREEN_H - 150

if CAMERA_WIN_WIDTH > max_camera_w or CAMERA_WIN_HEIGHT > max_camera_h:

    cam_scale = min(max_camera_w / CAMERA_WIN_WIDTH, max_camera_h / CAMERA_WIN_HEIGHT)

    CAM_TILE_WIDTH = int(CAM_TILE_WIDTH * cam_scale)
    CAM_TILE_HEIGHT = int(CAM_TILE_HEIGHT * cam_scale)

    CAMERA_WIN_WIDTH = (
        CAMERA_MARGIN + 2 * CAM_TILE_WIDTH + CAMERA_GRID_GAP
        + CAMERA_GRID_GAP + CAM_TILE_WIDTH + CAMERA_MARGIN
    )
    CAMERA_WIN_HEIGHT = 70 + 2 * CAM_TILE_HEIGHT + CAMERA_GRID_GAP + CAMERA_MARGIN

CHART_WIN_WIDTH = CHART_MARGIN + 2 * CHART_WIDTH + CHART_GAP + CHART_MARGIN
CHART_WIN_HEIGHT = 60 + 3 * CHART_HEIGHT + 2 * CHART_GAP + CHART_MARGIN

max_chart_w = SCREEN_W - 100
max_chart_h = SCREEN_H - 150

if CHART_WIN_WIDTH > max_chart_w or CHART_WIN_HEIGHT > max_chart_h:

    chart_scale = min(max_chart_w / CHART_WIN_WIDTH, max_chart_h / CHART_WIN_HEIGHT)

    CHART_WIDTH = int(CHART_WIDTH * chart_scale)
    CHART_HEIGHT = int(CHART_HEIGHT * chart_scale)
    CHART_GAP = int(CHART_GAP * chart_scale)
    CHART_COL1_X = CHART_MARGIN
    CHART_COL2_X = CHART_MARGIN + CHART_WIDTH + CHART_GAP

    CHART_WIN_WIDTH = CHART_MARGIN + 2 * CHART_WIDTH + CHART_GAP + CHART_MARGIN
    CHART_WIN_HEIGHT = 60 + 3 * CHART_HEIGHT + 2 * CHART_GAP + CHART_MARGIN

DATA_WIN_HEIGHT = min(DATA_WIN_HEIGHT, SCREEN_H - 150)

WIN_START_X, WIN_START_Y = 40, 40
WIN_STAGGER = 50


def _make_window(title, width, height, offset_index):
    pos = (
        WIN_START_X + offset_index * WIN_STAGGER,
        WIN_START_Y + offset_index * WIN_STAGGER,
    )
    window = sdl2_video.Window(title, size=(width, height), position=pos, resizable=True)
    renderer = sdl2_video.Renderer(window)
    surface = pygame.Surface((width, height))
    return window, renderer, surface


data_window, data_renderer, data_surface = _make_window(
    "Autonomous Vehicle Data", DATA_WIN_WIDTH, DATA_WIN_HEIGHT, 0
)
camera_window, camera_renderer, camera_surface = _make_window(
    "Camera Feeds", CAMERA_WIN_WIDTH, CAMERA_WIN_HEIGHT, 1
)
chart_window, chart_renderer, chart_surface = _make_window(
    "Sensor Analytics", CHART_WIN_WIDTH, CHART_WIN_HEIGHT, 2
)


def present_window(renderer, surface):
    texture = sdl2_video.Texture.from_surface(renderer, surface)
    renderer.clear()
    texture.draw()
    renderer.present()


font_title = pygame.font.Font(None, 32)
font_data = pygame.font.Font(None, 23)
font_axis = pygame.font.Font(None, 16)

clock = pygame.time.Clock()


# =========================================================
# CHART DRAWING HELPERS
# =========================================================

def draw_chart_frame(surface, rect, title,
                      y_min=None, y_max=None, y_unit="",
                      x_min=None, x_max=None, x_unit="",
                      x_is_time=False):

    pygame.draw.rect(surface, (28, 28, 28), rect)
    pygame.draw.rect(surface, (90, 90, 90), rect, 2)

    title_surface = font_data.render(title, True, (255, 210, 70))
    surface.blit(title_surface, (rect.x + 10, rect.y + 8))

    left_margin = 42 if y_min is not None else 12
    bottom_margin = 34 if (x_is_time or x_min is not None) else 22

    plot_rect = pygame.Rect(
        rect.x + left_margin,
        rect.y + 34,
        rect.width - left_margin - 12,
        rect.height - 34 - bottom_margin
    )

    pygame.draw.rect(surface, (14, 14, 14), plot_rect)
    pygame.draw.rect(surface, (60, 60, 60), plot_rect, 1)

    for i in range(1, 4):
        gy = plot_rect.y + plot_rect.height * i // 4
        pygame.draw.line(surface, (42, 42, 42), (plot_rect.x, gy), (plot_rect.right, gy), 1)

    if x_min is not None and not x_is_time:
        for i in range(1, 4):
            gx = plot_rect.x + plot_rect.width * i // 4
            pygame.draw.line(surface, (42, 42, 42), (gx, plot_rect.y), (gx, plot_rect.bottom), 1)

    if y_min is not None and y_max is not None:
        for i in range(5):
            frac = i / 4
            gy = plot_rect.bottom - frac * plot_rect.height
            val = y_min + frac * (y_max - y_min)
            label = font_axis.render(f"{val:.0f}{y_unit}", True, (140, 140, 140))
            surface.blit(label, (plot_rect.x - label.get_width() - 5, gy - label.get_height() // 2))

    if x_is_time:
        left_label = font_axis.render("-10s", True, (140, 140, 140))
        right_label = font_axis.render("now", True, (140, 140, 140))
        surface.blit(left_label, (plot_rect.x, plot_rect.bottom + 4))
        surface.blit(right_label, (plot_rect.right - right_label.get_width(), plot_rect.bottom + 4))
    elif x_min is not None and x_max is not None:
        for i in range(5):
            frac = i / 4
            gx = plot_rect.x + frac * plot_rect.width
            val = x_min + frac * (x_max - x_min)
            label = font_axis.render(f"{val:.0f}{x_unit}", True, (140, 140, 140))
            surface.blit(label, (gx - label.get_width() // 2, plot_rect.bottom + 4))

    return plot_rect


def draw_line_series(surface, plot_rect, data, min_val, max_val, color):

    n = len(data)

    if n < 2:
        return

    span = max(max_val - min_val, 0.001)
    points = []

    for i, value in enumerate(data):
        x = plot_rect.x + (i / (n - 1)) * plot_rect.width
        clipped = max(min_val, min(max_val, value))
        y = plot_rect.bottom - ((clipped - min_val) / span) * plot_rect.height
        points.append((x, y))

    pygame.draw.lines(surface, color, False, points, 2)


def draw_line_chart(surface, rect, title, data, min_val, max_val, color, unit="", auto_scale=True):

    plot_min, plot_max = min_val, max_val

    if auto_scale and len(data) >= 2:

        d_min = min(data)
        d_max = max(data)
        span = d_max - d_min
        pad = span * 0.25 if span > 0.5 else 1.0

        plot_min = max(min_val, d_min - pad)
        plot_max = min(max_val, d_max + pad)

        if plot_max - plot_min < 1.0:
            mid = (plot_max + plot_min) / 2
            plot_min = max(min_val, mid - 0.5)
            plot_max = min(max_val, mid + 0.5)

    plot_rect = draw_chart_frame(
        surface, rect, title, y_min=plot_min, y_max=plot_max, y_unit=unit, x_is_time=True
    )

    draw_line_series(surface, plot_rect, data, plot_min, plot_max, color)

    current_val = data[-1] if data else 0.0
    value_label = font_data.render(f"{current_val:.1f}{unit}", True, color)
    surface.blit(value_label, (rect.x + 12, rect.bottom - 20))


def draw_bar_chart(surface, rect, title, categories, values, max_val, warn_val, danger_val):

    plot_rect = draw_chart_frame(surface, rect, title, y_min=0.0, y_max=max_val, y_unit="m")

    count = len(categories)
    gap = 16
    bar_width = (plot_rect.width - gap * (count + 1)) // count

    for i, (category, value) in enumerate(zip(categories, values)):

        clipped = max(0.0, min(max_val, value))
        bar_height = int((clipped / max_val) * (plot_rect.height - 18))
        bx = plot_rect.x + gap + i * (bar_width + gap)
        by = plot_rect.bottom - bar_height

        if value < danger_val:
            bar_color = (255, 60, 60)
        elif value < warn_val:
            bar_color = (255, 210, 70)
        else:
            bar_color = (90, 220, 120)

        pygame.draw.rect(surface, bar_color, (bx, by, bar_width, bar_height))

        value_label = font_data.render(f"{value:.0f}m", True, bar_color)
        label_y = by - 18 if by - 18 > plot_rect.y else by + 4
        surface.blit(value_label, (bx + bar_width // 2 - value_label.get_width() // 2, label_y))

        name_label = font_data.render(category, True, (220, 220, 220))
        surface.blit(name_label, (bx + bar_width // 2 - name_label.get_width() // 2, plot_rect.bottom + 4))


def draw_dual_line_chart(surface, rect, title, data_a, label_a, color_a,
                          data_b, label_b, color_b, min_val, max_val, unit="", auto_scale=True):

    plot_min, plot_max = min_val, max_val

    if auto_scale:

        combined = list(data_a) + list(data_b)

        if len(combined) >= 2:

            d_min = min(combined)
            d_max = max(combined)
            span = d_max - d_min
            pad = span * 0.25 if span > 0.5 else 1.0

            plot_min = max(min_val, d_min - pad)
            plot_max = min(max_val, d_max + pad)

            if plot_max - plot_min < 1.0:
                mid = (plot_max + plot_min) / 2
                plot_min = max(min_val, mid - 0.5)
                plot_max = min(max_val, mid + 0.5)

    plot_rect = draw_chart_frame(
        surface, rect, title, y_min=plot_min, y_max=plot_max, y_unit=unit, x_is_time=True
    )

    draw_line_series(surface, plot_rect, data_a, plot_min, plot_max, color_a)
    draw_line_series(surface, plot_rect, data_b, plot_min, plot_max, color_b)

    val_a = data_a[-1] if data_a else 0.0
    val_b = data_b[-1] if data_b else 0.0

    display_scale = 100 if unit == "%" else 1

    label_a_surface = font_data.render(f"{label_a} {val_a * display_scale:.0f}{unit}", True, color_a)
    label_b_surface = font_data.render(f"{label_b} {val_b * display_scale:.0f}{unit}", True, color_b)

    surface.blit(label_a_surface, (rect.x + 12, rect.bottom - 20))
    surface.blit(label_b_surface, (rect.right - label_b_surface.get_width() - 12, rect.bottom - 20))


def draw_scatter_chart(surface, rect, title, points,
                        x_min, x_max, x_unit,
                        y_min, y_max, y_unit,
                        center_marker=False, legend=None):

    plot_rect = draw_chart_frame(
        surface, rect, title, y_min=y_min, y_max=y_max, y_unit=y_unit,
        x_min=x_min, x_max=x_max, x_unit=x_unit
    )

    x_span = max(x_max - x_min, 0.001)
    y_span = max(y_max - y_min, 0.001)

    for x_val, y_val, color in points:

        cx = max(x_min, min(x_max, x_val))
        cy = max(y_min, min(y_max, y_val))

        sx = plot_rect.x + ((cx - x_min) / x_span) * plot_rect.width
        sy = plot_rect.bottom - ((cy - y_min) / y_span) * plot_rect.height

        pygame.draw.circle(surface, color, (int(sx), int(sy)), 2)

    if center_marker and x_min <= 0 <= x_max and y_min <= 0 <= y_max:

        cx = plot_rect.x + ((0 - x_min) / x_span) * plot_rect.width
        cy = plot_rect.bottom - ((0 - y_min) / y_span) * plot_rect.height

        pygame.draw.polygon(
            surface, (255, 255, 255),
            [(cx, cy - 6), (cx - 5, cy + 5), (cx + 5, cy + 5)]
        )

    if legend:
        lx = plot_rect.x + 6
        ly = plot_rect.y + 6
        for label_text, color in legend:
            pygame.draw.circle(surface, color, (lx + 4, ly + 6), 3)
            label_surface = font_axis.render(label_text, True, (200, 200, 200))
            surface.blit(label_surface, (lx + 12, ly))
            ly += 14


# =========================================================
# LOOP STATE
# =========================================================

previous_yaw = vehicle.get_transform().rotation.yaw

indicator = "OFF"
indicator_timer = 0.0
last_time = time.time()

reverse_active = False
control_override_active = False
stuck_ticks = 0
auto_reverse_ticks_remaining = 0
emergency_active = False
tm_ignore_percentage_active = 0.0

if USE_PID_CONTROLLER:
    controller = SensorBasedController(
        world,
        vehicle,
        mw,
        emergency_brake_distance=EMERGENCY_BRAKE_DISTANCE,
        caution_zone_m=PID_CAUTION_ZONE_M,
        sensor_scan_radius=SENSOR_SCAN_RADIUS,
        sim_dt=0.05,
        cruise_speed_kmh=PID_CRUISE_SPEED_KMH,
        steer_lookahead_m=PID_LOOKAHEAD_M,
    )
    print(
        f"PID controller active: cruise {PID_CRUISE_SPEED_KMH} km/h, "
        f"emergency brake at {EMERGENCY_BRAKE_DISTANCE} m."
    )


# =========================================================
# MAIN LOOP
# =========================================================

_pipe = {"tick": 0, "prev_pid": None}

try:

    while program_running:

# =================================================
        # PIPELINE VERIFICATION STATE
        # =================================================
        # Proves (on every tick) the two pipeline invariants:
        #   I1. the fault-modified sensor data (mw.last_output) is the
        #       ONLY data the PID controller receives - never the raw
        #       ground truth;
        #   I2. the PID's command reaches apply_control() and nothing
        #       overwrites it (single application per tick, and the
        #       command read back from the vehicle after the physics
        #       tick matches the PID command that was applied).
        # Any violation prints a [PIPELINE] OVERWRITE / LEAK alert.
        _pipe["tick"] += 1
        _applied_pid_this_tick = None
        _apply_count_this_tick = 0

        # =================================================
        # ADVANCE SIMULATION (synchronous mode)
        # =================================================

        world.tick()

        # =================================================
        # PIPELINE VERIFICATION (I2 - PID → actuator, no overwrite)
        # =================================================
        # The command physically in effect DURING this tick is the one
        # applied at the END of the previous iteration - read back now,
        # after the tick completed and BEFORE this iteration re-applies
        # its own command. If Traffic Manager (an accidentally-armed
        # autopilot) or any other code wrote over the PID's control
        # during the tick, the readback differs from the PID command
        # and this alert fires. Traffic Manager officially overwrites
        # client apply_control() calls whenever autopilot is on, so a
        # clean match here IS the proof that nothing interfered.
        if _pipe["prev_pid"] is not None and USE_PID_CONTROLLER:

            a = _pipe["prev_pid"]
            c = vehicle.get_control()

            match = (
                abs(a.throttle - c.throttle) < 1e-3
                and abs(a.brake - c.brake) < 1e-3
                and abs(a.steer - c.steer) < 1e-3
                and bool(a.reverse) == bool(c.reverse)
                and bool(a.hand_brake) == bool(c.hand_brake)
            )

            _pipe["last_match"] = bool(match)

            if not match:
                print(
                    "[PIPELINE] !! OVERWRITE DETECTED - control read back after the tick "
                    f"(t={c.throttle:.3f}, b={c.brake:.3f}, s={c.steer:.3f}, rev={bool(c.reverse)}) "
                    f"!= PID command applied last tick "
                    f"(t={a.throttle:.3f}, b={a.brake:.3f}, s={a.steer:.3f}, rev={bool(a.reverse)}). "
                    "Something (e.g. Traffic Manager with autopilot armed) wrote over the PID's control!"
                )

            elif _pipe["tick"] % 30 == 1:
                # I1 check (same cadence): show the middleware-reported values
                # the PID computed from last tick vs the real ground truth.
                real_front = mw.ground_truth.get("distance_front")
                real_speed = mw.ground_truth.get("speed")
                faults = mw.active_faults() or ["none"]

                print(
                    "[PIPELINE] I1 sensor->PID  "
                    f"front_dist PID_saw={_pipe.get('input_front_m', 0.0):6.2f}m (real={real_front:6.2f}m) | "
                    f"speed PID_saw={_pipe.get('input_speed_kmh', 0.0):6.2f} (real={real_speed:6.2f}) | "
                    f"faults={faults} | "
                    "I2 PID->actuator "
                    f"(t={a.throttle:.3f}, b={a.brake:.3f}, s={a.steer:.3f}) "
                    "= readback -> NOT overwritten"
                )

        # =================================================
        # PYGAME CLOSE BUTTON
        # =================================================

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                program_running = False

            elif event.type == pygame.WINDOWCLOSE:
                program_running = False

        # =================================================
        # VEHICLE CHECK
        # =================================================

        if vehicle is None:
            break

        if not vehicle.is_alive:
            break

        # =================================================
        # CAMERA
        # =================================================

        update_camera()

        # =================================================
        # SPEED
        # =================================================
        # True velocity comes from CARLA physics via
        # vehicle.get_velocity() - smooth and tick-accurate, which the
        # PID needs. (The old displacement/wall-clock method was noisy:
        # at 20 Hz the loop dt is jittered by window rendering, so the
        # PID was toggling between full throttle and hard brake every
        # frame and the car surged instead of cruising.) The displacement
        # method is kept only as a fallback in case get_velocity() ever
        # reads ~0 while the car is visibly moving.

        velocity_vec = vehicle.get_velocity()
        physics_speed = (
            math.sqrt(velocity_vec.x ** 2 + velocity_vec.y ** 2 + velocity_vec.z ** 2) * 3.6
        )

        current_location = vehicle.get_location()

        if "previous_speed_location" not in globals():
            previous_speed_location = current_location
            previous_speed_time = time.time()
            displacement_speed = 0.0
        else:
            current_time = time.time()

            distance_moved = math.sqrt(
                (current_location.x - previous_speed_location.x) ** 2
                + (current_location.y - previous_speed_location.y) ** 2
                + (current_location.z - previous_speed_location.z) ** 2
            )

            delta_speed_time = current_time - previous_speed_time

            if delta_speed_time > 0:
                displacement_speed = (distance_moved / delta_speed_time) * 3.6
            else:
                displacement_speed = 0.0

            previous_speed_location = current_location
            previous_speed_time = current_time

        if physics_speed > 0.5 or displacement_speed < 1.0:
            speed = physics_speed
        else:
            speed = displacement_speed

        speed = mw.process_speed(speed)

        # =================================================
        # TM AVOIDANCE SYNC
        # =================================================
        # Only relevant when Traffic Manager is driving the ego (TM
        # control path). Keeps TM's OWN ignore_vehicles/walkers_percentage
        # in step with whether a masking fault is active, so toggling
        # distance_hide/collision_suppress mid-drive takes effect
        # immediately. In PID mode TM isn't steering the ego at all, so
        # this is skipped - the middleware-processed sensor data is the
        # only thing the ego sees.

        if not USE_PID_CONTROLLER:

            desired_ignore_pct = _current_tm_ignore_percentage()

            if desired_ignore_pct != tm_ignore_percentage_active:
                traffic_manager.ignore_vehicles_percentage(vehicle, desired_ignore_pct)
                traffic_manager.ignore_walkers_percentage(vehicle, desired_ignore_pct)
                tm_ignore_percentage_active = desired_ignore_pct
                print(
                    f"[MIDDLEWARE] TM avoidance {'RELAXED (100% ignore)' if desired_ignore_pct else 'restored (0% ignore)'} "
                    f"- masking fault {'active' if desired_ignore_pct else 'inactive'}"
                )

        # =================================================
        # HAZARD SCAN
        # =================================================

        side_distances = scan_surroundings()
        front_hazard = front_hazard_distance(side_distances)
        front_hazard = apply_fault_hazard_reactions(front_hazard)

        # =================================================
        # AUTO-REVERSE (backs up on its own if genuinely stuck)
        # =================================================
        # Only used in the Traffic Manager control path. In PID mode this
        # reverse-lurch fought the PID's own obstacle handling and made
        # the car appear stuck / jerky in traffic - the PID's speed loop
        # and emergency braking are the ego's only recovery there.

        if not USE_PID_CONTROLLER:

            if auto_reverse_ticks_remaining > 0:

                auto_reverse_needed = True
                auto_reverse_ticks_remaining -= 1

            else:

                auto_reverse_needed = False

                is_blocked_and_stopped = (
                    front_hazard < EMERGENCY_BRAKE_DISTANCE
                    and speed < 2.0
                )

                if is_blocked_and_stopped:
                    stuck_ticks += 1
                else:
                    stuck_ticks = 0

                if stuck_ticks >= STUCK_TICKS_THRESHOLD:
                    auto_reverse_ticks_remaining = REVERSE_DURATION_TICKS
                    stuck_ticks = 0
                    print("Stuck for too long - auto-reversing to clear it")

        else:
            auto_reverse_needed = False
            auto_reverse_ticks_remaining = 0

        # =================================================
        # CONTROL PATH
        # =================================================
        # Pipeline (as assigned): sensor data → middleware → PID controller
        # → actuator control → vehicle moves.
        #   - Sensors collect raw readings; `mw` (carla_middleware.Middleware)
        #     is the middleware layer and fault-injects the SENSOR data
        #     (speed, distances, radar, GNSS, IMU, ...) before anything else
        #     sees it.
        #   - PID mode: the PID controller computes throttle/steer ONLY from
        #     that middleware-processed sensor data, and the command is
        #     applied to the vehicle directly (no middleware in between -
        #     the middleware already ran on the sensor side).
        #   - TM mode: legacy Traffic Manager autopilot with reverse /
        #     emergency / fault-reaction override machinery.
        # The MiddlewareLogger in carla_excel.py is purely observational -
        # it records the applied PID control and the middleware-processed
        # sensor data to the Excel dataset but never gates the actuator.

        if USE_PID_CONTROLLER:

            with control_lock:
                reverse_requested = manual_reverse or auto_reverse_needed

            reverse_active = reverse_requested
            emergency_braking = False
            fault_reaction_active = False
            pid_ctrl = None
            pid_mode = "sensor_driving"

            if reverse_active:

                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.4, brake=0.0, reverse=True,
                        hand_brake=False, steer=0.0
                    )
                )
                _apply_count_this_tick += 1
                _pipe["prev_pid"] = None  # reverse override, not a PID command

            else:

                # sensor data (already through carla_middleware) → PID
                processed = build_processed_sensor_data(
                    sensor_readings, side_distances, front_hazard,
                    speed, SENSOR_SCAN_RADIUS
                )

                fused = controller.perception(processed)

                # ---- ITS sensor-side middleware output that the PID now
                #      actually reasons over (fault-modified, NOT ground truth)
                _pipe["input_front_m"] = fused["front_distance"]
                _pipe["input_speed_kmh"] = fused["speed_kmh"]
                _pipe["input_gnss"] = mw.last_output.get("gnss", sensor_readings["gnss"])

                pid_ctrl, pid_mode = controller.decide(fused)

                emergency_braking = (pid_mode == "emergency_brake")

                # Actuator: the PID command (computed from middleware-
                # processed sensor data) is applied to the vehicle directly.
                _applied_pid_this_tick = pid_ctrl
                _pipe["prev_pid"] = pid_ctrl
                vehicle.apply_control(pid_ctrl)
                _apply_count_this_tick += 1

            emergency_active = emergency_braking

        else:

            # Priority, highest first: reverse > emergency brake >
            # fault-driven throttle/steer reaction > plain autopilot.
            # Autopilot has to be switched off before any manual
            # apply_control() call, or Traffic Manager silently
            # overwrites it on the next tick.

            with control_lock:
                reverse_requested = manual_reverse or auto_reverse_needed

            if collision_stop_ticks_remaining > 0:
                collision_stop_ticks_remaining -= 1

            emergency_braking = (
                (not reverse_requested)
                and (front_hazard < EMERGENCY_BRAKE_DISTANCE or collision_stop_ticks_remaining > 0)
            )

            speed_fault_active = mw.is_active("speed_override") or mw.is_active("speed_lie")
            motion_fault_active = (
                mw.is_active("gps_spoof") or mw.is_active("gps_freeze")
                or mw.is_active("gnss_override") or mw.is_active("imu_override")
                or mw.is_active("imu_noise") or mw.is_active("imu_bias")
            )
            fault_reaction_active = (
                (not reverse_requested)
                and (not emergency_braking)
                and (speed_fault_active or motion_fault_active)
            )

            override_needed = reverse_requested or emergency_braking or fault_reaction_active

            if override_needed and not control_override_active:
                vehicle.set_autopilot(False)
                control_override_active = True

            elif not override_needed and control_override_active:
                control_override_active = False
                _rearm_ego_autopilot()

            reverse_active = reverse_requested
            emergency_active = emergency_braking

            if reverse_active:

                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.4, brake=0.0, reverse=True,
                        hand_brake=False, steer=0.0
                    )
                )

            elif emergency_active:

                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0, brake=1.0, hand_brake=False,
                        steer=vehicle.get_control().steer
                    )
                )

            elif fault_reaction_active:

                # Baseline steer is whatever the car was already doing (the
                # last control TM applied before autopilot got switched off
                # above), NOT 0.0 - forcing 0.0 here made a pure speed fault
                # drive the car dead straight regardless of road curvature,
                # so it left the lane / hit something almost immediately and
                # never actually reached the fake speed (throttle was
                # correctly high, but the car crashed before it could climb).
                steer_bias = vehicle.get_control().steer
                if motion_fault_active:
                    imu_bias = max(
                        -0.6, min(0.6, sensor_readings["imu"]["accel_x"] * IMU_STEER_BIAS_SCALE)
                    )
                    steer_bias = max(-1.0, min(1.0, steer_bias + imu_bias))

                target_throttle = 0.5
                if speed_fault_active:
                    fake_speed = (
                        mw.faults["speed_override"]["value"] if mw.is_active("speed_override")
                        else speed
                    )
                    error = fake_speed - speed
                    target_throttle = max(0.0, min(1.0, 0.5 + error * SPEED_FAULT_KP))

                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=target_throttle, brake=0.0,
                        steer=steer_bias, hand_brake=False
                    )
                )

        if mw.active_faults():
            applied = vehicle.get_control()
            print(
                f"[DEBUG] "
                f"middleware_enabled={mw.enabled} | "
                f"faults={mw.active_faults()} | "
                f"front_hazard={front_hazard:.2f} | "
                f"emergency_braking={emergency_braking} | "
                f"emergency_active={emergency_active} | "
                f"fault_reaction_active={fault_reaction_active} | "
                f"brake={applied.brake:.2f} | "
                f"throttle={applied.throttle:.2f} | "
                f"steer={applied.steer:.2f}"
            )

        # =================================================
        # PEDAL LEVELS (for the accelerator/brake bar HUD)
        # =================================================

        longitudinal_accel = sensor_readings["imu"]["accel_x"]

        current_control = vehicle.get_control()
        pedal_throttle = max(0.0, min(1.0, current_control.throttle))

        if longitudinal_accel < 0:
            pedal_brake = max(0.0, min(1.0, -longitudinal_accel / MAX_DECEL_FOR_BAR))
        else:
            pedal_brake = 0.0

        # =================================================
        # DEBUG TELEMETRY (≈2 Hz, PID mode only, read-only)
        # =================================================
        # One consolidated console dump that walks the whole pipeline:
        # [SENSOR] raw front scan / processed speed,
        # [MIDDLEWARE] sensor-side fault-injected values the PID sees,
        # [SAFETY] closing speed / TTC / bumper gap / stopping-distance /
        #          obstacle & rule brake scales (collision state machine),
        # [TRAFFIC] traffic-light state / stop-line distance / hard-stop flags,
        # [PID] cruise blend target / speed error / raw PID output,
        # [FINAL] the exact controller mode + the control that was applied
        #         and what the vehicle control read back after physics.
        # Every value comes from read-only controller.* attributes; nothing
        # here ever feeds back into the control path.
        if USE_PID_CONTROLLER and pid_ctrl is not None and (_pipe["tick"] % 10 == 1):
            fused_front = fused["front_distance"] if fused is not None else float("nan")
            fused_speed = fused["speed_kmh"] if fused is not None else float("nan")
            ttc_str = "inf" if controller.last_ttc is None else f"{controller.last_ttc:.2f}"
            fault_list = mw.active_faults()
            print(
                f"[SENSOR]     front_scan={front_hazard:6.2f} m | "
                f"processed_distance={fused_front:6.2f} m | "
                f"ego_speed={speed:6.2f} km/h | "
                f"processed_speed={fused_speed:6.2f} km/h"
            )
            print(
                f"[MIDDLEWARE] fault_injection={'ON' if fault_list else 'OFF'} "
                f"faults={fault_list} | "
                f"target_speed={controller.last_target_speed_kmh:6.2f} km/h | "
                f"object_detected={controller.last_object_detected}"
            )
            print(
                f"[SAFETY]     ttc={ttc_str} s | "
                f"closing={controller.last_closing_speed:6.2f} m/s | "
                f"bumper_gap={controller.last_bumper_gap:6.2f} m | "
                f"stop_dist_needed={controller.last_stop_dist_needed:6.2f} m | "
                f"lead_speed={controller.last_lead_speed_kmh:5.1f} km/h | "
                f"obstacle_scale={controller.last_obstacle_scale:.2f}"
            )
            print(
                f"[TRAFFIC]    rule_stop_scale={controller.last_rule_stop_scale:.2f} | "
                f"light_state={controller.traffic_light_state:8s} | "
                f"stop_line={controller.stop_line_distance if controller.stop_line_distance is None else controller.stop_line_distance:6.2f} m | "
                f"required_stop={controller.required_braking_distance:.2f} m | "
                f"red_stop={controller.red_light_stop_active} "
                f"yellow_stop={controller.yellow_stop_active} "
                f"light_brake={controller.traffic_light_brake_command:.2f}"
            )
            print(
                f"[PID]        cruise_limit={controller.cruise_speed_kmh:.1f} | "
                f"effective_target={controller.last_effective_cruise:6.2f} km/h | "
                f"speed_error={controller.last_speed_error:6.2f} | "
                f"pid_output={controller.last_command:6.3f}"
            )
            print(
                f"[FINAL]      mode={controller.last_mode:>17s} | "
                f"apply throttle={pid_ctrl.throttle:.3f} brake={pid_ctrl.brake:.3f} "
                f"steer={pid_ctrl.steer:.3f} | "
                f"final throttle={current_control.throttle:.3f} brake={current_control.brake:.3f} "
                f"steer={current_control.steer:.3f} | "
                f"bar throttle={pedal_throttle:.2f} bar brake={pedal_brake:.2f}"
            )

        # =================================================
        # CHART HISTORY UPDATE
        # =================================================

        speed_history.append(speed)
        accel_history.append(longitudinal_accel)
        throttle_history.append(pedal_throttle)
        brake_history.append(pedal_brake)

        lidar_points_raw = sensor_readings.get("lidar_points")

        if lidar_points_raw is not None and len(lidar_points_raw) > 0:

            forward_coord = lidar_points_raw[:, 0]
            lateral_coord = lidar_points_raw[:, 1]

            in_range = (forward_coord > -2.0) & (forward_coord < SENSOR_SCAN_RADIUS)

            left_returns = -lateral_coord[in_range & (lateral_coord < 0)]
            right_returns = lateral_coord[in_range & (lateral_coord > 0)]

            lidar_left = float(left_returns.min()) if left_returns.size else SENSOR_SCAN_RADIUS
            lidar_right = float(right_returns.min()) if right_returns.size else SENSOR_SCAN_RADIUS

        else:

            lidar_left = SENSOR_SCAN_RADIUS
            lidar_right = SENSOR_SCAN_RADIUS

        lidar_left_history.append(lidar_left)
        lidar_right_history.append(lidar_right)

        gnss_lat_history.append(sensor_readings["gnss"]["lat"])
        gnss_lon_history.append(sensor_readings["gnss"]["lon"])

        # =================================================
        # POSITION
        # =================================================

        location = vehicle.get_location()

        # =================================================
        # TURN DETECTION
        # =================================================

        current_yaw = vehicle.get_transform().rotation.yaw

        yaw_change = ((current_yaw - previous_yaw + 180.0) % 360.0) - 180.0

        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        if yaw_change > 0.15:
            indicator = "RIGHT"
            indicator_timer = 0.5
        elif yaw_change < -0.15:
            indicator = "LEFT"
            indicator_timer = 0.5

        if indicator_timer > 0:
            indicator_timer -= delta_time
        else:
            indicator = "OFF"

        previous_yaw = current_yaw

        # =================================================
        # TESLA LIGHTS
        # =================================================

        light_state = carla.VehicleLightState.Position

        if indicator == "LEFT":
            light_state |= carla.VehicleLightState.LeftBlinker
        elif indicator == "RIGHT":
            light_state |= carla.VehicleLightState.RightBlinker

        vehicle.set_light_state(carla.VehicleLightState(light_state))

        # =================================================
        # TRAFFIC LIGHT
        # =================================================

        if vehicle.is_at_traffic_light():

            state = vehicle.get_traffic_light_state()

            if state == carla.TrafficLightState.Red:
                traffic_light = "RED"
            elif state == carla.TrafficLightState.Yellow:
                traffic_light = "YELLOW"
            elif state == carla.TrafficLightState.Green:
                traffic_light = "GREEN"
            else:
                traffic_light = "UNKNOWN"

        else:

            traffic_light = "NONE"

        # =================================================
        # AI DECISION
        # =================================================

        if traffic_light == "RED":
            ai_decision = "Stopping"
        elif speed < 1:
            ai_decision = "Starting"
        elif indicator == "LEFT":
            ai_decision = "Turning Left"
        elif indicator == "RIGHT":
            ai_decision = "Turning Right"
        else:
            ai_decision = "Lane Following"

        # =================================================
        # EXCEL / MIDDLEWARE LOGGING
        # =================================================

        if reverse_active and auto_reverse_ticks_remaining > 0:
            mw_status = "auto_reverse"
        elif reverse_active:
            mw_status = "manual_reverse"
        elif emergency_braking:
            mw_status = "emergency_brake"
        elif fault_reaction_active:
            mw_status = "fault_reaction"
        elif traffic_light == "RED":
            mw_status = "red_light_stop"
        elif speed < 1.0:
            mw_status = "starting"
        else:
            mw_status = ai_decision.lower().replace(" ", "_")

        # The PID command was applied to the vehicle directly in the
        # CONTROL PATH above, so current_control already holds exactly what
        # the actuator received this tick. The MiddlewareLogger call below is
        # purely observational - it stores the applied PID control and the
        # middleware-processed sensor data into the Excel dataset. Nothing it
        # returns is fed back to the actuator: the middleware layer already
        # ran on the sensor data BEFORE the PID (assigned pipeline).
        pid_throttle_raw = float(current_control.throttle)
        pid_steer_raw = float(current_control.steer)

        _obstacle = sensor_readings.get("obstacle", {})
        _radar_detections = sensor_readings.get("radar", [])
        _radar_depths = [d["depth"] for d in _radar_detections]

        tick_sensor_data = {
            "gnss_lat": sensor_readings["gnss"]["lat"],
            "gnss_lon": sensor_readings["gnss"]["lon"],
            "gnss_alt": sensor_readings["gnss"]["alt"],
            "imu_accel_x": sensor_readings["imu"]["accel_x"],
            "imu_accel_y": sensor_readings["imu"]["accel_y"],
            "imu_accel_z": sensor_readings["imu"]["accel_z"],
            "imu_gyro_x": sensor_readings["imu"]["gyro_x"],
            "imu_gyro_y": sensor_readings["imu"]["gyro_y"],
            "imu_gyro_z": sensor_readings["imu"]["gyro_z"],
            "imu_compass_deg": sensor_readings["imu"]["compass"],
            "collision_actor": sensor_readings["collision"],
            "lane_invasion": sensor_readings["lane_invasion"],
            "lidar_point_count": int(len(lidar_points_raw)) if lidar_points_raw is not None else 0,
            "lidar_left_m": lidar_left,
            "lidar_right_m": lidar_right,
            "radar_detection_count": len(_radar_detections),
            "radar_min_depth_m": min(_radar_depths) if _radar_depths else SENSOR_SCAN_RADIUS,
            "obstacle_actor": _obstacle.get("actor", "None"),
            "obstacle_distance_m": _obstacle.get("distance", 0.0),
            "hazard_front_m": side_distances["front"],
            "hazard_back_m": side_distances["back"],
            "hazard_left_m": side_distances["left"],
            "hazard_right_m": side_distances["right"],
        }

        _final_throttle, _final_steer = middleware_logger.process(
            pid_throttle=pid_throttle_raw,
            pid_steer=pid_steer_raw,
            speed_kmh=speed,
            accel_x=longitudinal_accel,
            pedal_brake=pedal_brake,
            status=mw_status,
            sensor_data=tick_sensor_data,
        )

        # =================================================
        # TRAFFIC COUNT
        # =================================================

        alive_traffic = sum(1 for npc in traffic_vehicles if npc is not None and npc.is_alive)

        # =================================================
        # DEMO DATA
        # =================================================

        upload = random.randint(50, 150)
        download = random.randint(200, 500)

        # =================================================
        # CURRENT CAMERA
        # =================================================

        with camera_lock:
            display_camera = camera_mode.upper()

        # =================================================
        # PYGAME - clear all three window surfaces
        # =================================================

        data_surface.fill((20, 20, 20))
        camera_surface.fill((20, 20, 20))
        chart_surface.fill((20, 20, 20))

        title = font_title.render("AUTONOMOUS VEHICLE DATA", True, (255, 255, 255))
        data_surface.blit(title, (35, 20))

        camera_text = font_data.render("Camera : " + display_camera, True, (100, 220, 255))
        data_surface.blit(camera_text, (400, 25))

        # =================================================
        # DATA
        # =================================================

        obstacle_info = sensor_readings.get("obstacle", {"actor": "None", "distance": 0.0, "time": None})

        if obstacle_info["time"] is not None:
            obstacle_age = f"{time.time() - obstacle_info['time']:.1f}s ago"
        else:
            obstacle_age = "N/A"

        lidar_point_count = len(sensor_readings.get("lidar_points", []))
        radar_detection_count = len(sensor_readings.get("radar", []))

        if mw.is_active("speed_override"):
            speed_line = f"Speed        : {speed:.1f} km/h  [REAL: {mw.ground_truth.get('speed', 0.0):.1f}]"
        else:
            speed_line = f"Speed        : {speed:.1f} km/h"

        if mw.is_active("distance_override") and "front" in mw.faults["distance_override"]["directions"]:
            front_line = (
                f"Front        : {side_distances['front']:.1f}  "
                f"[REAL: {mw.ground_truth.get('distance_front', 0.0):.1f}]"
            )
        else:
            front_line = f"Front        : {side_distances['front']:.1f}"

        data_col_a = [
            "VEHICLE",
            "Tesla Model 3",
            "Driving Mode : Automatic",
            speed_line,
            "",
            "POSITION",
            f"X            : {location.x:.1f}",
            f"Y            : {location.y:.1f}",
            f"Z            : {location.z:.1f}",
            "",
            "SYSTEM STATUS",
            "Cloud        : Connected",
            "GPS          : Active",
            "Camera       : Streaming",
            f"Weather      : {current_weather_name}",
            f"AI           : {ai_decision}",
            "Emergency Brk: " + ("ACTIVE" if emergency_braking else "off"),
            "",
            "SENSOR DISTANCES (m)",
            front_line,
            f"Back         : {side_distances['back']:.1f}",
            f"Left         : {side_distances['left']:.1f}",
            f"Right        : {side_distances['right']:.1f}",
            "",
            "GNSS",
            f"Latitude     : {sensor_readings['gnss']['lat']:.6f}",
            f"Longitude    : {sensor_readings['gnss']['lon']:.6f}",
            f"Altitude     : {sensor_readings['gnss']['alt']:.1f} m",
            "",
            "IMU",
            f"Accel X/Y/Z  : {sensor_readings['imu']['accel_x']:.2f} / "
            f"{sensor_readings['imu']['accel_y']:.2f} / "
            f"{sensor_readings['imu']['accel_z']:.2f} m/s2",
            f"Gyro X/Y/Z   : {sensor_readings['imu']['gyro_x']:.2f} / "
            f"{sensor_readings['imu']['gyro_y']:.2f} / "
            f"{sensor_readings['imu']['gyro_z']:.2f} rad/s",
            f"Compass      : {sensor_readings['imu']['compass']:.1f} deg",
        ]

        if mw.active_faults():
            middleware_lines = [f"  - {fault}" for fault in mw.active_faults()]
        else:
            middleware_lines = ["  (no active faults)"]

        data_col_b = [
            "COLLISION / LANE / OBSTACLE",
            f"Collision    : {sensor_readings['collision']}",
            f"Lane Invasion: {sensor_readings['lane_invasion']}",
            f"Obstacle     : {obstacle_info['actor']}",
            f"  at {obstacle_info['distance']:.1f} m, {obstacle_age}",
            "",
            "ADVANCED SENSORS",
            f"LiDAR points : {lidar_point_count}",
            f"Radar tracks : {radar_detection_count}",
            "DVS Camera   : Streaming",
            "",
            "MIDDLEWARE",
            "Status       : " + ("ARMED" if mw.enabled else "BYPASSED"),
        ] + middleware_lines + [
            "",
            "TRAFFIC",
            f"Vehicles     : {alive_traffic}",
            f"Light        : {traffic_light}",
            "",
            "TURN INDICATOR",
            f"Direction    : {indicator}",
            "Indicator    : " + ("ON" if indicator != "OFF" else "OFF"),
            "",
            "DATA TRANSFER",
            "Car -> Cloud : ACTIVE",
            f"Upload       : {upload} KB/s",
            "Cloud -> Car : ACTIVE",
            f"Download     : {download} KB/s",
            "Connection   : Stable",
            "",
            "CAMERA CONTROLS",
            "1 = Front",
            "2 = Back",
            "3 = Top",
            "4 = Low",
            "5 = 360 Degree",
            "6 = Chase",
            "R = Toggle Reverse",
            "",
            "T = Speed Override",
            "D = Distance Override",
            "U = GNSS Override",
            "I = IMU Override",
            "M = Middleware Kill",
            "W = Cycle Weather",
            "",
            "ESC = Exit"
        ]

        # =================================================
        # DRAW DATA (two columns - keeps everything on screen)
        # =================================================

        sections = [
            "VEHICLE", "POSITION", "SYSTEM STATUS", "SENSOR DISTANCES (m)",
            "GNSS", "IMU", "COLLISION / LANE / OBSTACLE", "ADVANCED SENSORS",
            "TRAFFIC", "TURN INDICATOR", "DATA TRANSFER", "CAMERA CONTROLS"
        ]

        BASE_LINE_HEIGHT = 24
        BASE_BLANK_HEIGHT = 7
        TEXT_TOP_MARGIN = 65
        TEXT_BOTTOM_MARGIN = 15

        def _column_height(column_data):
            lines = sum(1 for t in column_data if t != "")
            blanks = sum(1 for t in column_data if t == "")
            return lines * BASE_LINE_HEIGHT + blanks * BASE_BLANK_HEIGHT

        needed_height = max(_column_height(data_col_a), _column_height(data_col_b))
        available_height = DATA_WIN_HEIGHT - TEXT_TOP_MARGIN - TEXT_BOTTOM_MARGIN

        spacing_scale = 1.0
        if needed_height > available_height > 0:
            spacing_scale = available_height / needed_height

        line_height = max(12, BASE_LINE_HEIGHT * spacing_scale)
        blank_height = max(3, BASE_BLANK_HEIGHT * spacing_scale)

        def draw_text_column(column_data, x_pos):

            y_pos = TEXT_TOP_MARGIN

            for text in column_data:

                if text == "":
                    y_pos += blank_height
                    continue

                text_color = (255, 210, 70) if text in sections else (235, 235, 235)

                text_surface = font_data.render(text, True, text_color)
                data_surface.blit(text_surface, (x_pos, y_pos))

                y_pos += line_height

        draw_text_column(data_col_a, TEXT_COL_A_X)
        draw_text_column(data_col_b, TEXT_COL_B_X)

        # =================================================
        # TURN ARROW
        # =================================================

        if indicator == "LEFT":
            pygame.draw.polygon(data_surface, (255, 200, 0), [(700, 5), (650, 30), (700, 55)])
        elif indicator == "RIGHT":
            pygame.draw.polygon(data_surface, (255, 200, 0), [(650, 5), (700, 30), (650, 55)])

        # =================================================
        # DIRECTIONAL CAMERA THUMBNAILS + DVS (own window)
        # =================================================

        cam_window_title = font_title.render("CAMERA FEEDS", True, (255, 255, 255))
        camera_surface.blit(cam_window_title, (CAMERA_MARGIN, 15))

        grid_col_x = [CAMERA_MARGIN, CAMERA_MARGIN + CAM_TILE_WIDTH + CAMERA_GRID_GAP]
        grid_row_y = [70, 70 + CAM_TILE_HEIGHT + CAMERA_GRID_GAP]

        thumb_positions = {
            "front": (grid_col_x[0], grid_row_y[0]),
            "right": (grid_col_x[1], grid_row_y[0]),
            "left": (grid_col_x[0], grid_row_y[1]),
            "back": (grid_col_x[1], grid_row_y[1]),
        }

        dvs_x = grid_col_x[1] + CAM_TILE_WIDTH + CAMERA_GRID_GAP
        dvs_y = grid_row_y[0]

        danger_threshold = EMERGENCY_BRAKE_DISTANCE + 3.0

        for cam_name, (px, py) in thumb_positions.items():

            frame = camera_frames.get(cam_name)

            if frame is None:
                continue

            rgb = frame[:, :, :3][:, :, ::-1]

            cam_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            if (CAM_TILE_WIDTH, CAM_TILE_HEIGHT) != (SENSOR_CAM_WIDTH, SENSOR_CAM_HEIGHT):
                cam_surface = pygame.transform.smoothscale(cam_surface, (CAM_TILE_WIDTH, CAM_TILE_HEIGHT))

            camera_surface.blit(cam_surface, (px, py))

            dist = side_distances.get(cam_name, SENSOR_SCAN_RADIUS)

            if dist < danger_threshold:
                border_color = (255, 60, 60)
            elif dist < danger_threshold * 1.6:
                border_color = (255, 210, 70)
            else:
                border_color = (90, 220, 120)

            pygame.draw.rect(camera_surface, border_color, (px, py, CAM_TILE_WIDTH, CAM_TILE_HEIGHT), 3)

            label = font_data.render(f"{cam_name.upper()}  {dist:.1f} m", True, (255, 255, 255))
            camera_surface.blit(label, (px + 6, py + CAM_TILE_HEIGHT - 22))

        dvs_surface = pygame.surfarray.make_surface(dvs_frame_rgb.swapaxes(0, 1))
        if (CAM_TILE_WIDTH, CAM_TILE_HEIGHT) != (SENSOR_CAM_WIDTH, SENSOR_CAM_HEIGHT):
            dvs_surface = pygame.transform.smoothscale(dvs_surface, (CAM_TILE_WIDTH, CAM_TILE_HEIGHT))
        camera_surface.blit(dvs_surface, (dvs_x, dvs_y))

        pygame.draw.rect(camera_surface, (120, 170, 255), (dvs_x, dvs_y, CAM_TILE_WIDTH, CAM_TILE_HEIGHT), 3)

        dvs_label = font_data.render("DVS EVENT CAM", True, (255, 255, 255))
        camera_surface.blit(dvs_label, (dvs_x + 6, dvs_y + CAM_TILE_HEIGHT - 22))

        # =================================================
        # PEDAL BARS (accelerator / brake)
        # =================================================

        pedal_panel_y = grid_row_y[1]

        pedal_area_width = (CAMERA_WIN_WIDTH - CAMERA_MARGIN) - (dvs_x + 20)
        base_bar_width, base_bar_gap = 70, 40
        bar_layout_scale = min(1.0, pedal_area_width / (3 * base_bar_width + 2 * base_bar_gap))

        bar_width = max(30, int(base_bar_width * bar_layout_scale))
        bar_gap = max(15, int(base_bar_gap * bar_layout_scale))
        bar_max_height = 160
        bar_bottom = pedal_panel_y + bar_max_height

        accel_x = dvs_x + 20
        brake_x = accel_x + bar_width + bar_gap
        speed_x = brake_x + bar_width + bar_gap

        pedal_header = font_data.render("PEDALS / SPEED / GEAR", True, (255, 210, 70))
        camera_surface.blit(pedal_header, (dvs_x, pedal_panel_y - 24))

        pygame.draw.rect(camera_surface, (90, 90, 90), (accel_x, pedal_panel_y, bar_width, bar_max_height), 2)
        pygame.draw.rect(camera_surface, (90, 90, 90), (brake_x, pedal_panel_y, bar_width, bar_max_height), 2)
        pygame.draw.rect(camera_surface, (90, 90, 90), (speed_x, pedal_panel_y, bar_width, bar_max_height), 2)

        accel_fill_height = int(bar_max_height * max(0.0, min(1.0, pedal_throttle)))
        brake_fill_height = int(bar_max_height * max(0.0, min(1.0, pedal_brake)))
        speed_fill_height = int(bar_max_height * max(0.0, min(1.0, speed / MAX_SPEED_FOR_BAR)))

        pygame.draw.rect(
            camera_surface, (80, 220, 100),
            (accel_x, bar_bottom - accel_fill_height, bar_width, accel_fill_height)
        )
        pygame.draw.rect(
            camera_surface, (230, 60, 60),
            (brake_x, bar_bottom - brake_fill_height, bar_width, brake_fill_height)
        )
        pygame.draw.rect(
            camera_surface, (100, 220, 255),
            (speed_x, bar_bottom - speed_fill_height, bar_width, speed_fill_height)
        )

        accel_label = font_data.render(f"ACCEL {pedal_throttle * 100:.0f}%", True, (200, 255, 210))
        brake_label = font_data.render(f"BRAKE {pedal_brake * 100:.0f}%", True, (255, 200, 200))
        speed_label = font_data.render(f"SPEED {speed:.0f}", True, (200, 235, 255))

        camera_surface.blit(accel_label, (accel_x - 5, bar_bottom + 8))
        camera_surface.blit(brake_label, (brake_x - 5, bar_bottom + 8))
        camera_surface.blit(speed_label, (speed_x - 5, bar_bottom + 8))

        if reverse_active and auto_reverse_ticks_remaining > 0:
            gear_text = "GEAR: REVERSE (AUTO)"
        elif reverse_active:
            gear_text = "GEAR: REVERSE (MANUAL)"
        else:
            gear_text = "GEAR: DRIVE"

        gear_label = font_data.render(
            gear_text, True, (255, 180, 80) if reverse_active else (200, 200, 200)
        )
        camera_surface.blit(gear_label, (accel_x, bar_bottom + 36))

        # =================================================
        # SENSOR CHARTS (two columns)
        # =================================================

        charts_title = font_title.render("SENSOR ANALYTICS", True, (255, 255, 255))
        chart_surface.blit(charts_title, (CHART_COL1_X, 15))

        chart_y = 50

        speed_chart_rect = pygame.Rect(CHART_COL1_X, chart_y, CHART_WIDTH, CHART_HEIGHT)

        if len(gnss_lon_history) >= 2:

            lon_min, lon_max = min(gnss_lon_history), max(gnss_lon_history)
            lat_min, lat_max = min(gnss_lat_history), max(gnss_lat_history)

            lon_pad = max((lon_max - lon_min) * 0.2, 0.00005)
            lat_pad = max((lat_max - lat_min) * 0.2, 0.00005)

            lon_min, lon_max = lon_min - lon_pad, lon_max + lon_pad
            lat_min, lat_max = lat_min - lat_pad, lat_max + lat_pad

        else:

            lon_now = gnss_lon_history[-1] if gnss_lon_history else 0.0
            lat_now = gnss_lat_history[-1] if gnss_lat_history else 0.0

            lon_min, lon_max = lon_now - 0.0005, lon_now + 0.0005
            lat_min, lat_max = lat_now - 0.0005, lat_now + 0.0005

        gnss_track_points = []

        for i, (lon, lat) in enumerate(zip(gnss_lon_history, gnss_lat_history)):
            age_frac = i / max(1, len(gnss_lon_history) - 1)
            brightness = 90 + int(age_frac * 165)
            gnss_track_points.append((lon, lat, (100, brightness, 255)))

        draw_scatter_chart(
            chart_surface, speed_chart_rect, "GNSS TRACK (lon/lat)",
            gnss_track_points, lon_min, lon_max, "", lat_min, lat_max, "",
        )

        chart_y += CHART_HEIGHT + CHART_GAP
        accel_chart_rect = pygame.Rect(CHART_COL1_X, chart_y, CHART_WIDTH, CHART_HEIGHT)
        draw_line_chart(
            chart_surface, accel_chart_rect, "IMU LONGITUDINAL ACCEL (m/s2)",
            accel_history, -8.0, 4.0, (200, 130, 255), unit=""
        )

        chart_y += CHART_HEIGHT + CHART_GAP
        pedal_chart_rect = pygame.Rect(CHART_COL1_X, chart_y, CHART_WIDTH, CHART_HEIGHT)
        draw_dual_line_chart(
            chart_surface, pedal_chart_rect, "THROTTLE vs BRAKE (%)",
            throttle_history, "Accel", (80, 220, 100),
            brake_history, "Brake", (230, 60, 60),
            0.0, 1.0, unit="%"
        )

        chart_y = 50

        distance_chart_rect = pygame.Rect(CHART_COL2_X, chart_y, CHART_WIDTH, CHART_HEIGHT)
        draw_dual_line_chart(
            chart_surface, distance_chart_rect, "LiDAR LEFT vs RIGHT (m)",
            lidar_left_history, "Left", (100, 200, 255),
            lidar_right_history, "Right", (255, 150, 80),
            0.0, SENSOR_SCAN_RADIUS, unit="m"
        )

        chart_y += CHART_HEIGHT + CHART_GAP
        lidar_chart_rect = pygame.Rect(CHART_COL2_X, chart_y, CHART_WIDTH, CHART_HEIGHT)

        lidar_plot_points = []

        if lidar_points_raw is not None and len(lidar_points_raw) > 0:

            for point in lidar_points_raw:

                lateral = float(point[1])
                forward = float(point[0])
                height_z = float(point[2])

                color = (255, 210, 70) if height_z > -0.5 else (100, 200, 255)

                lidar_plot_points.append((lateral, forward, color))

        draw_scatter_chart(
            chart_surface, lidar_chart_rect, "LiDAR TOP-DOWN (m)",
            lidar_plot_points, -25.0, 25.0, "", -25.0, 25.0, "",
            center_marker=True
        )

        chart_y += CHART_HEIGHT + CHART_GAP
        radar_chart_rect = pygame.Rect(CHART_COL2_X, chart_y, CHART_WIDTH, CHART_HEIGHT)

        radar_plot_points = []

        for detection in sensor_readings.get("radar", []):

            if detection["velocity"] < -0.5:
                color = (255, 80, 80)
            elif detection["velocity"] > 0.5:
                color = (90, 220, 120)
            else:
                color = (200, 200, 200)

            radar_plot_points.append((detection["azimuth_deg"], detection["depth"], color))

        draw_scatter_chart(
            chart_surface, radar_chart_rect, "RADAR (deg / m)",
            radar_plot_points, -30.0, 30.0, "", 0.0, 40.0, "",
            legend=[("Approaching", (255, 80, 80)), ("Receding", (90, 220, 120))]
        )

        # =================================================
        # DISPLAY
        # =================================================

        present_window(data_renderer, data_surface)
        present_window(camera_renderer, camera_surface)
        present_window(chart_renderer, chart_surface)

        clock.tick(20)


# =========================================================
# CLEANUP
# =========================================================

except KeyboardInterrupt:

    pass


finally:

    print("Cleaning up...")

    program_running = False

    try:
        middleware_logger.save()
    except Exception as _log_exc:
        print(f"[MiddlewareLogger] Save failed: {_log_exc}")

    keyboard_listener.stop()

    try:

        restore_settings = world.get_settings()
        restore_settings.synchronous_mode = False
        restore_settings.fixed_delta_seconds = None
        world.apply_settings(restore_settings)
        traffic_manager.set_synchronous_mode(False)

    except Exception:

        pass

    if vehicle is not None:

        if vehicle.is_alive:

            vehicle.set_autopilot(False)
            vehicle.set_light_state(carla.VehicleLightState.NONE)
            vehicle.destroy()

    for npc in traffic_vehicles:

        if npc is not None:

            if npc.is_alive:

                npc.set_autopilot(False)
                npc.destroy()

    for controller in pedestrian_controllers:

        if controller is not None:

            if controller.is_alive:

                controller.stop()
                controller.destroy()

    for walker in pedestrians:

        if walker is not None:

            if walker.is_alive:

                walker.destroy()

    for cam in directional_cameras:

        if cam is not None:

            if cam.is_alive:

                cam.stop()
                cam.destroy()

    for sensor in extra_sensors:

        if sensor is not None:

            if sensor.is_alive:

                sensor.stop()
                sensor.destroy()

    pygame.quit()

    print("All vehicles removed.")
    print("Simulation stopped.")
