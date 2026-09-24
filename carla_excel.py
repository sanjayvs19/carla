import threading
import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment


# =========================================================
# MIDDLEWARE LOGGER  (Excel / sensor-logging layer)
# =========================================================
# Architecture (the professor's pipeline):
#     CARLA sensors → Middleware (carla_middleware.Middleware,
#                       fault-injects SENSOR data)
#                   → PID controller (examples/pid_controller)
#                   → Actuator (vehicle.apply_control) → CARLA
#                            ↓
#          MiddlewareLogger (this class, EXCEL / observational)
#
# The MiddlewareLogger is a purely OBSERVATIONAL logging layer.
# It does NOT sit between the PID and the actuator and it never
# modifies or gates the applied control.  Every tick it simply
# records the middleware-processed sensor data and the PID
# command that the main loop applied to the vehicle (so the
# Excel dataset always contains the true commanded values).
#
# Columns written now (operational):
#   timestamp, speed_kmh, accel_x, pedal_brake, steer,
#   pid_throttle, pid_steer,
#   middleware_throttle, middleware_steer,
#   status
#
# Columns reserved for future vulnerability research
# (always written, currently filled with sentinel values):
#   attack_type, attack_status,
#   pid_original_throttle, pid_original_steer,   ← immutable copies
#   mw_modified_throttle,  mw_modified_steer,
#   deviation_throttle,    deviation_steer,
#   final_throttle,        final_steer
# =========================================================

class MiddlewareLogger:
    """Central logging layer that intercepts PID→actuator data,
    records every tick to an in-memory buffer, and flushes the
    buffer to a timestamped .xlsx file on shutdown."""

    # ------------------------------------------------------------------
    # Column layout – order is the display order in the spreadsheet.
    # Changing a name here automatically propagates to the header row.
    # ------------------------------------------------------------------
    COLUMNS = [
        # --- Operational telemetry ---
        "timestamp",
        "speed_kmh",
        "accel_x_ms2",
        "pedal_brake",
        "steer",
        # --- PID layer outputs (as computed by Traffic Manager) ---
        "pid_throttle",
        "pid_steer",
        # --- Middleware layer outputs (after any modification) ---
        "middleware_throttle",
        "middleware_steer",
        # --- Operational status ---
        "status",
        # ── SENSOR SUITE COLUMNS ─────────────────────────────────────
        # One snapshot per tick from every sensor attached to the ego
        # vehicle, alongside the derived hazard-scan readout. These are
        # always populated (the sensors are always active), unlike the
        # vulnerability columns below.
        "gnss_lat", "gnss_lon", "gnss_alt",
        "imu_accel_x", "imu_accel_y", "imu_accel_z",
        "imu_gyro_x", "imu_gyro_y", "imu_gyro_z", "imu_compass_deg",
        "collision_actor",
        "lane_invasion",
        "lidar_point_count", "lidar_left_m", "lidar_right_m",
        "radar_detection_count", "radar_min_depth_m",
        "obstacle_actor", "obstacle_distance_m",
        "hazard_front_m", "hazard_back_m", "hazard_left_m", "hazard_right_m",
        # ── VULNERABILITY RESEARCH COLUMNS ──────────────────────────
        # These are always present so the schema never changes between
        # normal and attack runs.  During normal operation they hold
        # the sentinel values shown in _SENTINEL below.
        # INSERT ATTACK LOGIC in the process() method below where
        # indicated by the  ▼ ATTACK INJECTION POINT ▼  comment.
        "attack_type",          # e.g. "throttle_spike", "steer_offset"
        "attack_status",        # "none" | "active" | "blocked"
        "pid_original_throttle",# immutable copy of PID throttle, pre-attack
        "pid_original_steer",   # immutable copy of PID steer,    pre-attack
        "mw_modified_throttle", # value after attack modification (or same as original)
        "mw_modified_steer",    # value after attack modification (or same as original)
        "deviation_throttle",   # mw_modified_throttle - pid_original_throttle
        "deviation_steer",      # mw_modified_steer    - pid_original_steer
        "final_throttle",       # value actually sent to actuator
        "final_steer",          # value actually sent to actuator
    ]

    _SENTINEL_STR = "none"
    _SENTINEL_NUM = 0.0

    def __init__(self):
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f"simulation_log_{run_id}.xlsx"
        self._rows   = []          # in-memory buffer (list of dicts)
        self._lock   = threading.Lock()
        print(f"[MiddlewareLogger] Logging to: {self.filepath}")

    # ------------------------------------------------------------------
    # process() – called once per simulation tick.
    # Receives the raw PID values from the Traffic Manager, applies any
    # middleware logic (currently: pass-through only), logs everything,
    # and returns the (possibly modified) final values.
    # ------------------------------------------------------------------
    def process(
        self,
        pid_throttle: float,
        pid_steer:    float,
        speed_kmh:    float,
        accel_x:      float,
        pedal_brake:  float,
        status:       str,
        sensor_data:  dict = None,
    ) -> tuple:
        """
        Parameters
        ----------
        pid_throttle : throttle command produced by PID/Traffic Manager [0-1]
        pid_steer    : steer command produced by PID/Traffic Manager    [-1,1]
        speed_kmh    : current vehicle speed (km/h)
        accel_x      : IMU longitudinal acceleration (m/s²)
        pedal_brake  : brake level derived from deceleration            [0-1]
        status       : human-readable status string for this tick
        sensor_data  : optional dict of this tick's sensor-suite readings
                       (GNSS, IMU, collision, lane invasion, LiDAR, radar,
                       obstacle detector, hazard scan). Missing keys are
                       logged as sentinel values so the schema never
                       changes between calls.

        Returns
        -------
        (final_throttle, final_steer) – values to pass to actuators
        """

        sensor_data = sensor_data or {}

        # ── Immutable copies of what PID computed ─────────────────────
        # These are NEVER overwritten, even if an attack modifies the
        # outgoing values.  The original PID intent is always preserved
        # in the log for forensic comparison.
        original_throttle = float(pid_throttle)
        original_steer    = float(pid_steer)

        # ── Middleware pass-through (no modification yet) ──────────────
        mw_throttle = original_throttle
        mw_steer    = original_steer

        # ┌─────────────────────────────────────────────────────────────┐
        # │          ▼  ATTACK INJECTION POINT  ▼                      │
        # │                                                             │
        # │  Future vulnerability code goes here.  The attack receives  │
        # │  original_throttle / original_steer, produces modified      │
        # │  mw_throttle / mw_steer, and sets attack_type /            │
        # │  attack_status.  The original_* variables must NEVER be     │
        # │  reassigned – only mw_* may be changed.                    │
        # │                                                             │
        # │  Example skeleton (do NOT uncomment – for reference only): │
        # │                                                             │
        # │  if attack_enabled:                                         │
        # │      attack_type   = "throttle_spike"                      │
        # │      attack_status = "active"                               │
        # │      mw_throttle   = min(1.0, original_throttle + 0.3)     │
        # │      # original_throttle is intentionally left unchanged   │
        # └─────────────────────────────────────────────────────────────┘
        attack_type   = self._SENTINEL_STR
        attack_status = self._SENTINEL_STR

        # ── Final values sent to the actuator ─────────────────────────
        final_throttle = mw_throttle
        final_steer    = mw_steer

        # ── Deviation (zero during normal operation) ───────────────────
        deviation_throttle = final_throttle - original_throttle
        deviation_steer    = final_steer    - original_steer

        # ── Build log row ──────────────────────────────────────────────
        row = {
            "timestamp":              datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "speed_kmh":              round(speed_kmh, 3),
            "accel_x_ms2":            round(accel_x, 4),
            "pedal_brake":            round(pedal_brake, 4),
            "steer":                  round(original_steer, 4),
            "pid_throttle":           round(original_throttle, 4),
            "pid_steer":              round(original_steer, 4),
            "middleware_throttle":    round(mw_throttle, 4),
            "middleware_steer":       round(mw_steer, 4),
            "status":                 status,
            # sensor suite columns
            "gnss_lat":               round(sensor_data.get("gnss_lat", self._SENTINEL_NUM), 6),
            "gnss_lon":               round(sensor_data.get("gnss_lon", self._SENTINEL_NUM), 6),
            "gnss_alt":               round(sensor_data.get("gnss_alt", self._SENTINEL_NUM), 2),
            "imu_accel_x":            round(sensor_data.get("imu_accel_x", self._SENTINEL_NUM), 4),
            "imu_accel_y":            round(sensor_data.get("imu_accel_y", self._SENTINEL_NUM), 4),
            "imu_accel_z":            round(sensor_data.get("imu_accel_z", self._SENTINEL_NUM), 4),
            "imu_gyro_x":             round(sensor_data.get("imu_gyro_x", self._SENTINEL_NUM), 4),
            "imu_gyro_y":             round(sensor_data.get("imu_gyro_y", self._SENTINEL_NUM), 4),
            "imu_gyro_z":             round(sensor_data.get("imu_gyro_z", self._SENTINEL_NUM), 4),
            "imu_compass_deg":        round(sensor_data.get("imu_compass_deg", self._SENTINEL_NUM), 2),
            "collision_actor":        sensor_data.get("collision_actor", self._SENTINEL_STR),
            "lane_invasion":          sensor_data.get("lane_invasion", self._SENTINEL_STR),
            "lidar_point_count":      sensor_data.get("lidar_point_count", 0),
            "lidar_left_m":           round(sensor_data.get("lidar_left_m", self._SENTINEL_NUM), 3),
            "lidar_right_m":          round(sensor_data.get("lidar_right_m", self._SENTINEL_NUM), 3),
            "radar_detection_count":  sensor_data.get("radar_detection_count", 0),
            "radar_min_depth_m":      round(sensor_data.get("radar_min_depth_m", self._SENTINEL_NUM), 3),
            "obstacle_actor":         sensor_data.get("obstacle_actor", self._SENTINEL_STR),
            "obstacle_distance_m":    round(sensor_data.get("obstacle_distance_m", self._SENTINEL_NUM), 3),
            "hazard_front_m":         round(sensor_data.get("hazard_front_m", self._SENTINEL_NUM), 3),
            "hazard_back_m":          round(sensor_data.get("hazard_back_m", self._SENTINEL_NUM), 3),
            "hazard_left_m":          round(sensor_data.get("hazard_left_m", self._SENTINEL_NUM), 3),
            "hazard_right_m":         round(sensor_data.get("hazard_right_m", self._SENTINEL_NUM), 3),
            # vulnerability columns
            "attack_type":            attack_type,
            "attack_status":          attack_status,
            "pid_original_throttle":  round(original_throttle, 4),
            "pid_original_steer":     round(original_steer, 4),
            "mw_modified_throttle":   round(mw_throttle, 4),
            "mw_modified_steer":      round(mw_steer, 4),
            "deviation_throttle":     round(deviation_throttle, 6),
            "deviation_steer":        round(deviation_steer, 6),
            "final_throttle":         round(final_throttle, 4),
            "final_steer":            round(final_steer, 4),
        }

        with self._lock:
            self._rows.append(row)

        return final_throttle, final_steer

    # ------------------------------------------------------------------
    # save() – call once when the simulation ends.
    # Writes the in-memory buffer to a well-formatted .xlsx file.
    # ------------------------------------------------------------------
    def save(self):
        """Flush all buffered rows to disk as a formatted .xlsx file."""
        with self._lock:
            rows_snapshot = list(self._rows)

        if not rows_snapshot:
            print("[MiddlewareLogger] No data to save.")
            return

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Simulation Log"

        # ── Styles ────────────────────────────────────────────────────
        header_font   = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        data_font     = Font(name="Arial", size=9)
        op_fill       = PatternFill("solid", fgColor="1F4E79")   # dark blue  – operational cols
        sensor_fill   = PatternFill("solid", fgColor="2E5E3E")   # dark green – sensor suite cols
        vuln_fill     = PatternFill("solid", fgColor="7B2C2C")   # dark red   – vulnerability cols
        center_align  = Alignment(horizontal="center", vertical="center", wrap_text=True)

        SENSOR_COLS = {
            "gnss_lat", "gnss_lon", "gnss_alt",
            "imu_accel_x", "imu_accel_y", "imu_accel_z",
            "imu_gyro_x", "imu_gyro_y", "imu_gyro_z", "imu_compass_deg",
            "collision_actor", "lane_invasion",
            "lidar_point_count", "lidar_left_m", "lidar_right_m",
            "radar_detection_count", "radar_min_depth_m",
            "obstacle_actor", "obstacle_distance_m",
            "hazard_front_m", "hazard_back_m", "hazard_left_m", "hazard_right_m",
        }

        VULNERABILITY_COLS = {
            "attack_type", "attack_status",
            "pid_original_throttle", "pid_original_steer",
            "mw_modified_throttle",  "mw_modified_steer",
            "deviation_throttle",    "deviation_steer",
            "final_throttle",        "final_steer",
        }

        # ── Header row ────────────────────────────────────────────────
        for col_idx, col_name in enumerate(self.COLUMNS, start=1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.font      = header_font
            if col_name in VULNERABILITY_COLS:
                cell.fill = vuln_fill
            elif col_name in SENSOR_COLS:
                cell.fill = sensor_fill
            else:
                cell.fill = op_fill
            cell.alignment = center_align

        # ── Data rows ─────────────────────────────────────────────────
        for row_idx, row_dict in enumerate(rows_snapshot, start=2):
            for col_idx, col_name in enumerate(self.COLUMNS, start=1):
                cell = ws.cell(row=row_idx, column=col_idx, value=row_dict.get(col_name, ""))
                cell.font      = data_font
                cell.alignment = Alignment(horizontal="center", vertical="center")

        # ── Column widths ─────────────────────────────────────────────
        col_widths = {
            "timestamp": 24, "status": 22, "attack_type": 18, "attack_status": 14,
            "collision_actor": 22, "lane_invasion": 22, "obstacle_actor": 22,
        }
        for col_idx, col_name in enumerate(self.COLUMNS, start=1):
            width = col_widths.get(col_name, 18)
            ws.column_dimensions[
                openpyxl.utils.get_column_letter(col_idx)
            ].width = width

        # ── Freeze header row ─────────────────────────────────────────
        ws.freeze_panes = "A2"

        # ── Auto-filter ───────────────────────────────────────────────
        ws.auto_filter.ref = ws.dimensions

        wb.save(self.filepath)
        print(
            f"[MiddlewareLogger] Saved {len(rows_snapshot):,} rows → {self.filepath}"
        )
