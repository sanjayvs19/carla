"""
Sensor-to-Controller Middleware for AV Cybersecurity Research
================================================================
Every sensor reading in the main script passes through one of the
process_*() methods below before the rest of the code (HUD, hazard
detection, emergency braking, auto-reverse) ever sees it. That makes
this the single choke point for the whole vehicle's perception -
exactly where a real in-vehicle bus attack (spoofed CAN frames,
GPS spoofing, sensor jamming) would sit.

WHY A SEPARATE MODULE
----------------------
Keeping this out of the main script means:
  - You can edit/add faults here without touching driving logic.
  - You can toggle faults live via hotkeys, or from a debugger, or
    from a second script, without restarting the sim.
  - The "attack surface" is in one file, which is exactly what you
    want for a vulnerability-simulation paper: one place to point to.

QUICK USE
---------
    from carla_middleware import Middleware
    mw = Middleware()

    # inside a sensor callback:
    lat, lon, alt = mw.process_gnss(data.latitude, data.longitude, data.altitude)

    # turn a fault on/off, anytime, from anywhere:
    mw.set_fault("gps_spoof", True, offset_lat=0.01, offset_lon=0.01)
    mw.toggle_fault("distance_hide")
    mw.enabled = False   # master kill-switch - guarantees clean passthrough

Every fault is just an entry in self.faults - add your own the same
way the existing ones are written, no other code needs to change.
"""

import random
import time

import numpy as np


class Middleware:

    def __init__(self):

        # Master switch. Flip this off and every process_*() call
        # below becomes a pure passthrough, regardless of individual
        # fault settings - useful as a big "disarm everything" button.
        self.enabled = True

        # ------------------------------------------------------------
        # FAULT LIBRARY - this is the part you edit/extend.
        # Each fault is: enabled flag + its own tunable parameters +
        # a human-readable label (shown on the HUD / in logs).
        # ------------------------------------------------------------
        self.faults = {

            "speed_override": {
                "enabled": False,
                "value": 80.0,          # km/h reported, regardless of real speed
                "label": "Speed Override (fixed fake value)",
            },

            "distance_override": {
                "enabled": False,
                "value": 2.0,           # meters reported, regardless of real distance -
                                         # below EMERGENCY_BRAKE_DISTANCE by default so
                                         # toggling this on demonstrates the vulnerability
                                         # immediately (phantom braking) instead of
                                         # silently reporting a "safe" 30m
                "directions": {"front"},
                "label": "Distance Override (fixed fake value)",
            },

            "gnss_override": {
                "enabled": False,
                "lat": 0.0,
                "lon": 0.0,
                "alt": 0.0,
                "label": "GNSS Override (fixed fake position)",
            },

            "imu_override": {
                "enabled": False,
                "accel_x": 0.0, "accel_y": 0.0, "accel_z": 0.0,
                "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
                "label": "IMU Override (fixed fake motion)",
            },

            "gps_spoof": {
                "enabled": False,
                "offset_lat": 0.01,
                "offset_lon": 0.01,
                "label": "GPS Spoofing",
            },

            "gps_freeze": {
                "enabled": False,
                "label": "GPS Freeze (stale position)",
            },

            "imu_noise": {
                "enabled": False,
                "accel_noise_std": 2.0,
                "gyro_noise_std": 0.5,
                "label": "IMU Noise Injection",
            },

            "imu_bias": {
                "enabled": False,
                "accel_bias_x": 3.0,
                "label": "IMU Bias / Drift",
            },

            "distance_shrink": {
                "enabled": False,
                "factor": 3.0,          # reports objects this many times closer than real
                "directions": {"front"},
                "label": "False-Near Obstacle (phantom braking)",
            },

            "distance_hide": {
                "enabled": False,
                "directions": {"front"},
                "label": "Obstacle Masking (hides real hazards)",
            },

            "speed_lie": {
                "enabled": False,
                "factor": 0.5,          # reports speed as this fraction of the real value
                "label": "Speedometer Falsification",
            },

            "lidar_dropout": {
                "enabled": False,
                "drop_fraction": 0.7,
                "label": "LiDAR Point Dropout",
            },

            "lidar_ghost": {
                "enabled": False,
                "count": 40,
                "label": "LiDAR Ghost Points",
            },

            "radar_ghost": {
                "enabled": False,
                "count": 3,
                "label": "Radar Ghost Targets",
            },

            "collision_suppress": {
                "enabled": False,
                "label": "Collision Alert Suppression",
            },
        }

        self._latched_gnss = None

        self.event_log = []

        # Real (uncorrupted) value seen most recently per channel, and
        # what was actually handed to the rest of the pipeline after
        # any fault ran. Lets the HUD show "reported X, real Y" side
        # by side whenever an override is live.
        self.ground_truth = {}
        self.last_output = {}

        self._log("Middleware initialised - all faults off, passthrough mode")

    # ------------------------------------------------------------
    # Control API - this is the "easy to access and change" part.
    # Call these from a hotkey, a console, another script, anywhere.
    # ------------------------------------------------------------

    def set_fault(self, name, enabled=True, **params):
        """Turn a fault on/off and optionally update its parameters
        in the same call, e.g.:
            mw.set_fault("gps_spoof", True, offset_lat=0.02)

        Only the enabled flag and whichever params are actually passed
        get touched - anything omitted keeps its previous value, and
        nothing here ever nudges a value back toward ground truth. This
        is also what the dashboard's /api/set_fault calls directly, so
        param values coming in as JSON (strings/lists from the browser)
        are coerced to match the type already stored for that key.
        """
        if name not in self.faults:
            raise KeyError(f"Unknown fault '{name}'. Available: {list(self.faults)}")

        self.faults[name]["enabled"] = enabled

        for key, value in params.items():
            current = self.faults[name].get(key)

            if isinstance(current, set):
                if isinstance(value, str):
                    value = {v.strip() for v in value.split(",") if v.strip()}
                elif isinstance(value, (list, tuple, set)):
                    value = {str(v).strip() for v in value if str(v).strip()}
            elif isinstance(current, bool):
                value = bool(value)
            elif isinstance(current, (int, float)):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    pass
            # unknown key (current is None) or a plain string param
            # (e.g. a future non-numeric field) - store as given.

            self.faults[name][key] = value

        state = "ENABLED" if enabled else "disabled"
        self._log(f"{state}: {self.faults[name]['label']}")

    def get_state(self):
        """JSON-serializable snapshot of every fault's enabled flag,
        label, and current parameters (sets become sorted lists). Pure
        read - never mutates anything, safe to call from the dashboard
        on every poll."""
        state = {}
        for name, f in self.faults.items():
            params = {}
            for key, value in f.items():
                if key in ("enabled", "label"):
                    continue
                params[key] = sorted(value) if isinstance(value, set) else value
            state[name] = {
                "enabled": f["enabled"],
                "label": f["label"],
                "params": params,
            }
        return {
            "master_enabled": self.enabled,
            "faults": state,
        }

    def toggle_fault(self, name):
        current = self.faults[name]["enabled"]
        self.set_fault(name, not current)

    def is_active(self, name):
        return self.enabled and self.faults[name]["enabled"]

    def active_faults(self):
        """Human-readable list of everything currently corrupting the
        data - handy for a HUD line or a paper's attack timeline."""
        if not self.enabled:
            return []
        return [f["label"] for f in self.faults.values() if f["enabled"]]

    def _log(self, message):
        self.event_log.append((time.time(), message))
        if len(self.event_log) > 200:
            self.event_log.pop(0)
        print(f"[MIDDLEWARE] {message}")

    # ------------------------------------------------------------
    # Sensor processors - one per sensor type. Each takes the real
    # reading in, and returns the (possibly corrupted) reading the
    # rest of the pipeline will actually act on.
    # ------------------------------------------------------------

    def process_gnss(self, lat, lon, alt):

        self.ground_truth["gnss"] = (lat, lon, alt)

        if not self.enabled:
            self.last_output["gnss"] = (lat, lon, alt)
            return lat, lon, alt

        if self.is_active("gnss_override"):
            f = self.faults["gnss_override"]
            result = (f["lat"], f["lon"], f["alt"])
            self.last_output["gnss"] = result
            return result

        if self.is_active("gps_freeze"):
            if self._latched_gnss is None:
                self._latched_gnss = (lat, lon, alt)
            self.last_output["gnss"] = self._latched_gnss
            return self._latched_gnss
        else:
            self._latched_gnss = None

        if self.is_active("gps_spoof"):
            f = self.faults["gps_spoof"]
            lat = lat + f["offset_lat"]
            lon = lon + f["offset_lon"]

        self.last_output["gnss"] = (lat, lon, alt)
        return lat, lon, alt

    def process_imu(self, accel_xyz, gyro_xyz, compass):
        """accel_xyz / gyro_xyz are (x, y, z) tuples."""

        self.ground_truth["imu"] = (accel_xyz, gyro_xyz, compass)

        if not self.enabled:
            self.last_output["imu"] = (accel_xyz, gyro_xyz, compass)
            return accel_xyz, gyro_xyz, compass

        if self.is_active("imu_override"):
            f = self.faults["imu_override"]
            result = (
                (f["accel_x"], f["accel_y"], f["accel_z"]),
                (f["gyro_x"], f["gyro_y"], f["gyro_z"]),
                compass
            )
            self.last_output["imu"] = result
            return result

        ax, ay, az = accel_xyz
        gx, gy, gz = gyro_xyz

        if self.is_active("imu_noise"):
            f = self.faults["imu_noise"]
            ax += random.gauss(0, f["accel_noise_std"])
            ay += random.gauss(0, f["accel_noise_std"])
            gx += random.gauss(0, f["gyro_noise_std"])
            gy += random.gauss(0, f["gyro_noise_std"])

        if self.is_active("imu_bias"):
            f = self.faults["imu_bias"]
            ax += f["accel_bias_x"]

        result = ((ax, ay, az), (gx, gy, gz), compass)
        self.last_output["imu"] = result
        return result

    def process_distance(self, direction, value, scan_radius):
        """direction: 'front' / 'back' / 'left' / 'right'. This feeds
        straight into emergency braking and auto-reverse, so it's the
        main lever for making the car brake for nothing (phantom
        braking) or fail to brake for something real (masked hazard)."""

        self.ground_truth[f"distance_{direction}"] = value

        if not self.enabled:
            self.last_output[f"distance_{direction}"] = value
            return value

        if (self.is_active("distance_override")
                and direction in self.faults["distance_override"]["directions"]):
            result = self.faults["distance_override"]["value"]
            self.last_output[f"distance_{direction}"] = result
            return result

        if (self.is_active("distance_hide")
                and direction in self.faults["distance_hide"]["directions"]):
            self.last_output[f"distance_{direction}"] = scan_radius
            return scan_radius   # tells the car "nothing there"

        if (self.is_active("distance_shrink")
                and direction in self.faults["distance_shrink"]["directions"]):
            factor = self.faults["distance_shrink"]["factor"]
            # A factor of 1.0 or less is a silent no-op for a "shrink"
            # fault (dividing by 1 leaves the value unchanged, and
            # anything under 1 would actually grow it) - both are easy
            # to enter from the dashboard's plain number input, which
            # has no floor. Force a minimum of 1.5 so the fault always
            # visibly shrinks the reading whenever it's switched on,
            # instead of silently doing nothing at factor=1.
            if not isinstance(factor, (int, float)) or factor <= 1.0:
                factor = 1.5
            result = max(0.1, value / factor)
            self.last_output[f"distance_{direction}"] = result
            return result

        self.last_output[f"distance_{direction}"] = value
        return value

    def process_speed(self, speed_kmh):

        self.ground_truth["speed"] = speed_kmh

        if not self.enabled:
            self.last_output["speed"] = speed_kmh
            return speed_kmh

        if self.is_active("speed_override"):
            result = self.faults["speed_override"]["value"]
            self.last_output["speed"] = result
            return result

        if self.is_active("speed_lie"):
            result = speed_kmh * self.faults["speed_lie"]["factor"]
            self.last_output["speed"] = result
            return result

        self.last_output["speed"] = speed_kmh
        return speed_kmh

    def process_lidar(self, points):
        """points: numpy array, shape (N, 4) -> (x, y, z, intensity)."""

        if not self.enabled or points is None or len(points) == 0:
            return points

        if self.is_active("lidar_dropout"):
            frac = self.faults["lidar_dropout"]["drop_fraction"]
            keep_mask = np.random.rand(len(points)) > frac
            points = points[keep_mask]

        if self.is_active("lidar_ghost"):
            n = self.faults["lidar_ghost"]["count"]
            dtype = points.dtype if len(points) else np.float32
            ghosts = np.zeros((n, 4), dtype=dtype)
            ghosts[:, 0] = np.random.uniform(2, 15, n)    # fake returns ahead
            ghosts[:, 1] = np.random.uniform(-3, 3, n)
            ghosts[:, 2] = np.random.uniform(-1, 1, n)
            points = np.vstack([points, ghosts]) if len(points) else ghosts

        return points

    def process_radar(self, detections):
        """detections: list of dicts with 'depth', 'azimuth_deg', 'velocity'."""

        if not self.enabled:
            return detections

        if self.is_active("radar_ghost"):
            n = self.faults["radar_ghost"]["count"]
            for _ in range(n):
                detections.append({
                    "depth": random.uniform(5, 20),
                    "azimuth_deg": random.uniform(-25, 25),
                    "velocity": random.uniform(-8, -1),  # fake fast-approaching target
                })

        return detections

    def process_collision(self, actor_name):

        if not self.enabled:
            return actor_name

        if self.is_active("collision_suppress"):
            return "None"

        return actor_name
