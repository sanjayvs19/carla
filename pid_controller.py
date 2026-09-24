"""
pid_controller.py
==================
The EGO Tesla's perception-fusion + lane tracking + PID controller.
"""

import math
import carla


def build_processed_sensor_data(sensor_readings, side_distances, front_hazard,
                                 speed_reported, sensor_scan_radius):
    """
    Builds the processed snapshot dict passed to controller.perception().
    """
    radar_detections = sensor_readings.get("radar", [])
    # Filter for dynamic approaching targets in forward cone
    approaching_depths = [
        d["depth"] for d in radar_detections
        if abs(d.get("azimuth_deg", 0.0)) < 20.0 and d.get("velocity", 0.0) < -0.5
    ]

    return {
        "front_distance": front_hazard,
        "back_distance": side_distances["back"],
        "left_distance": side_distances["left"],
        "right_distance": side_distances["right"],
        "speed_kmh": speed_reported,
        "imu": dict(sensor_readings["imu"]),
        "gnss": dict(sensor_readings["gnss"]),
        "lidar_point_count": (
            len(sensor_readings["lidar_points"])
            if sensor_readings.get("lidar_points") is not None else 0
        ),
        "radar_min_approaching_depth": min(approaching_depths) if approaching_depths else sensor_scan_radius,
        "collision_actor": sensor_readings["collision"],
    }


class SensorBasedController:
    """THE single control path for the EGO Tesla."""

    def __init__(
        self,
        world,
        vehicle,
        mw,
        *,
        emergency_brake_distance: float = 6.0,
        caution_zone_m: float = 6.0,
        sensor_scan_radius: float = 30.0,
        sim_dt: float = 0.05,

        # --- closing-speed / time-to-collision look-ahead ---
        # The ego reacts EARLY to a closing lead vehicle, not just when it
        # is already inside the static 5-8m band. Closing speed is derived
        # from the middleware-processed front distances (no radar sign
        # conventions involved). reaction_time extends the effective hazard
        # distance by how fast the gap is closing; the two TTC thresholds
        # scale braking and define emergency.
        reaction_time: float = 0.8,
        ttc_emergency: float = 1.5,    # seconds - below this = full emergency stop
        ttc_caution: float = 3.0,      # seconds - below this = start blending off throttle

        # --- speed-aware stopping-distance emergency ---
        # front_distance is measured CENTER-to-CENTER, so the bumper gap is
        # front minus the two half-lengths. If the gap is closing fast and
        # the ego can no longer brake to a stop inside that bumper gap at
        # the assumed max sustained decel, collision is otherwise
        # physically inevitable - force full emergency brake regardless of
        # the static distance / TTC thresholds. This catches a lead that
        # slams its brakes (TTC lags by design because it is derived from
        # distance deltas).
        emergency_decel_mps2: float = 4.5,
        vehicle_half_len_m: float = 2.3,

        # --- speed control loop (throttle/brake) ---
        cruise_speed_kmh: float = 30.0,
        speed_kp: float = 0.08,
        speed_ki: float = 0.02,
        speed_kd: float = 0.005,
        speed_integral_limit: float = 15.0,

        # --- steering control loop (lane-follow) ---
        steer_lookahead_m: float = 8.0,
        steer_kp: float = 1.2,
        steer_ki: float = 0.02,
        steer_kd: float = 0.05,
        steer_integral_limit: float = 0.3,
        # Lateral cross-track correction: steers back toward lane center.
        # Heading-only control drifts out of the lane; adding the signed
        # lateral offset (Stanley-style) keeps the car centered.
        steer_kp_cte: float = 0.30,

        # --- steering bias terms (obstacle avoidance + fault effects) ---
        steer_avoid_gain: float = 0.15,
        imu_steer_bias_scale: float = 0.05,
        gnss_steer_error_gain: float = 0.01,

        # --- D-term low-pass filter ---
        # The raw derivative multiplies per-tick error noise by 1/sim_dt
        # (x20 at 0.05s) before the KD gain scales it, which caused the
        # jerky/shaky throttle+steering symptom. A first-order exponential
        # filter smooths the D-term; alpha=1.0 disables it.
        derivative_filter_alpha: float = 0.5,

        # --- traffic-light hard-stop ---
        # A red light is a HARD stop (throttle 0 + brake), not a cruise
        # blend. Braking starts as soon as the distance to the stop line
        # drops to (required stopping distance + margin), where the
        # required distance comes from the middleware-processed speed and
        # this assumed comfort-brake deceleration. YELLOW stops only when a
        # safe stop before the line is still possible, otherwise proceeds.
        red_light_decel_mps2: float = 3.0,
        stop_line_margin_m: float = 2.0,
        stop_hold_brake: float = 0.25,
    ):
        self.world = world
        self.vehicle = vehicle
        self.mw = mw

        self.emergency_brake_distance = emergency_brake_distance
        self.caution_zone_m = caution_zone_m
        self.sensor_scan_radius = sensor_scan_radius
        self.sim_dt = sim_dt

        # Closing-speed / TTC look-ahead state
        self.reaction_time = reaction_time
        self.ttc_emergency = ttc_emergency
        self.ttc_caution = ttc_caution
        self._prev_front = None

        self.emergency_decel_mps2 = emergency_decel_mps2
        self.vehicle_half_len_m = vehicle_half_len_m

        self.cruise_speed_kmh = cruise_speed_kmh
        self.speed_kp = speed_kp
        self.speed_ki = speed_ki
        self.speed_kd = speed_kd
        self.speed_integral_limit = speed_integral_limit

        self.steer_lookahead_m = steer_lookahead_m
        self.steer_kp = steer_kp
        self.steer_ki = steer_ki
        self.steer_kd = steer_kd
        self.steer_integral_limit = steer_integral_limit
        self.steer_kp_cte = steer_kp_cte

        self.steer_avoid_gain = steer_avoid_gain
        self.imu_steer_bias_scale = imu_steer_bias_scale
        self.gnss_steer_error_gain = gnss_steer_error_gain
        self.derivative_filter_alpha = derivative_filter_alpha

        # Traffic-light hard-stop tuning
        self.red_light_decel_mps2 = red_light_decel_mps2
        self.stop_line_margin_m = stop_line_margin_m
        self.stop_hold_brake = stop_hold_brake

        # PID state
        self._speed_error_integral = 0.0
        self._speed_error_previous = None
        self._speed_error_deriv_f = None
        self._steer_error_integral = 0.0
        self._steer_error_previous = None
        self._steer_error_deriv_f = None

        # Persistent Lane / Waypoint Tracker (locks to ego's lane)
        self._tracked_wp = None

        # Last control-cycle telemetry (read by excel_data.py for the
        # [SENSOR]/[MIDDLEWARE]/[RULE]/[TTC]/[PID] debug output - pure
        # reads, never feed back into the control path).
        self.last_closing_speed = 0.0        # m/s, >0 = gap closing
        self.last_ttc = None                 # seconds, None when not closing
        self.last_object_detected = False    # anything in the front cone
        self.last_target_speed_kmh = 0.0     # estimated speed of object ahead
        self.last_effective_cruise = 0.0     # speed-PID target this cycle
        self.last_speed_error = 0.0          # effective_cruise - current
        self.last_command = 0.0              # raw PID output (pre clamps)
        self.last_mode = "sensor_driving"
        self.last_emergency = False

        # Traffic-light / stop-line telemetry (read by the [TRAFFIC] dump)
        self.traffic_light_state = "None"
        self.stop_line_distance = None
        self.required_braking_distance = 0.0
        self.red_light_stop_active = False
        self.yellow_stop_active = False
        self.traffic_light_brake_command = 0.0

        # Collision telemetry (read by the [SAFETY] dump)
        self.last_bumper_gap = 0.0
        self.last_stop_dist_needed = 0.0
        self.last_lead_speed_kmh = 0.0
        self.last_obstacle_scale = 0.0
        self.last_rule_stop_scale = 0.0

        # One-shot event edge detectors (print only on state change)
        self._in_emergency = False
        self._traffic_stop_was_active = False
        self._caution_was_active = False

    def perception(self, processed):
        # The lane-gated front cone is the authority on "something in my
        # path". Radar approaching-depth may only REFINE that reading (it
        # sees oncoming traffic in other lanes too and would otherwise
        # produce the same phantom braking the cone gate now prevents).
        front_cone = processed["front_distance"]
        if front_cone < self.sensor_scan_radius:
            fused_front = min(front_cone, processed["radar_min_approaching_depth"])
        else:
            fused_front = front_cone

        gnss_ground_truth = self.mw.ground_truth.get("gnss")
        gnss_reported = self.mw.last_output.get("gnss")
        gnss_error_m = 0.0

        if gnss_ground_truth and gnss_reported:
            d_lat = gnss_reported[0] - gnss_ground_truth[0]
            d_lon = gnss_reported[1] - gnss_ground_truth[1]
            meters_per_lat_deg = 111320.0
            meters_per_lon_deg = 111320.0 * math.cos(math.radians(gnss_ground_truth[0]))
            gnss_error_m = math.sqrt((d_lat * meters_per_lat_deg) ** 2 + (d_lon * meters_per_lon_deg) ** 2)

        return {
            "front_distance": fused_front,
            "left_distance": processed["left_distance"],
            "right_distance": processed["right_distance"],
            "speed_kmh": processed["speed_kmh"],
            "imu_accel_x": processed["imu"]["accel_x"],
            "gnss_error_m": gnss_error_m,
            "obstacle_ahead": fused_front < self.emergency_brake_distance,
            "obstacle_caution": fused_front < (self.emergency_brake_distance + self.caution_zone_m),
        }

    def compute_heading_error(self):
        """Maintains continuous lane tracking along the ego vehicle's lane.
        Advances waypoints along the same lane rather than snapping to adjacent lanes."""
        transform = self.vehicle.get_transform()
        ego_loc = transform.location
        forward = transform.get_forward_vector()

        # Initialize or recover tracked waypoint
        if self._tracked_wp is None:
            self._tracked_wp = self.world.get_map().get_waypoint(
                ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )

        # Advance tracked waypoint as the car moves forward along its lane
        while self._tracked_wp.transform.location.distance(ego_loc) < 4.0:
            next_wps = self._tracked_wp.next(3.0)
            if not next_wps:
                break
            if len(next_wps) == 1:
                self._tracked_wp = next_wps[0]
            else:
                # Junction branch: pick the branch continuing straight ahead
                def deviation(wp):
                    dx = wp.transform.location.x - ego_loc.x
                    dy = wp.transform.location.y - ego_loc.y
                    l = math.sqrt(dx**2 + dy**2)
                    return 1.0 - (forward.x * dx + forward.y * dy) / max(0.01, l)
                self._tracked_wp = min(next_wps, key=deviation)

        # Get target lookahead point along the current lane path
        lookahead_candidates = self._tracked_wp.next(self.steer_lookahead_m)
        if lookahead_candidates:
            if len(lookahead_candidates) == 1:
                target_wp = lookahead_candidates[0]
            else:
                def deviation(wp):
                    dx = wp.transform.location.x - ego_loc.x
                    dy = wp.transform.location.y - ego_loc.y
                    l = math.sqrt(dx**2 + dy**2)
                    return 1.0 - (forward.x * dx + forward.y * dy) / max(0.01, l)
                target_wp = min(lookahead_candidates, key=deviation)
        else:
            target_wp = self._tracked_wp

        target = target_wp.transform.location
        to_target_x = target.x - ego_loc.x
        to_target_y = target.y - ego_loc.y
        to_target_len = math.sqrt(to_target_x ** 2 + to_target_y ** 2)

        if to_target_len < 0.01:
            return 0.0

        dot = forward.x * to_target_x + forward.y * to_target_y
        cross = forward.x * to_target_y - forward.y * to_target_x

        return math.atan2(cross, dot)

    def compute_cross_track_error(self):
        """Signed lateral offset (meters) of the ego from its lane centerline.
        Positive = ego to the LEFT of the lane center, negative = right.
        Used to steer back toward the center of the lane."""
        transform = self.vehicle.get_transform()
        ego_loc = transform.location
        forward = transform.get_forward_vector()

        lane_wp = self.world.get_map().get_waypoint(
            ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        lane_loc = lane_wp.transform.location

        dx = ego_loc.x - lane_loc.x
        dy = ego_loc.y - lane_loc.y

        return forward.x * dy - forward.y * dx

    def compute_red_stop_scale(self):
        """Backwards-compatible wrapper over compute_traffic_light_stop().
        Returns 1.0 while a hard red/yellow stop is active, else 0.0."""
        mode = self.compute_traffic_light_stop(getattr(self, "_last_stop_speed_kmh", 0.0) or 0.0)
        return (0.0 if mode == "proceed" else 1.0)

    def compute_traffic_light_stop(self, speed_kmh):
        """Hard red/yellow traffic-light stop using CARLA stop-line
        waypoints (0.9.16: light.get_stop_waypoints()).

        Returns 'red_stop' / 'yellow_stop' when the ego MUST brake this
        tick (throttle 0 + progressive brake), else 'proceed'. Sets the
        [TRAFFIC] telemetry attributes along the way.

        - Only the stop line belonging to the EGO'S OWN lane (same
          road_id + lane_id) is used, so a red light for a perpendicular
          or opposite approach never stops this car (TEST 8).
        - Red: brake as soon as distance-to-line <= required stopping
          distance + margin; required distance comes from the
          middleware-processed speed (never ground truth).
        - Yellow: safe stopping rule - stop iff the ego can still stop
          before the line, otherwise proceed (TEST 0/yellow).
        - Red holds the car stopped (small brake) until the light is green.
        """
        self._last_stop_speed_kmh = speed_kmh
        self.traffic_light_state = "None"
        self.stop_line_distance = None
        self.required_braking_distance = 0.0
        self.red_light_stop_active = False
        self.yellow_stop_active = False
        self.traffic_light_brake_command = 0.0
        try:
            state = self.vehicle.get_traffic_light_state()
            light = self.vehicle.get_traffic_light()
            if light is None:
                return "proceed"
            self.traffic_light_state = str(state).split(".")[-1]  # e.g. "Red"

            v_mps = max(0.0, speed_kmh / 3.6)
            self.required_braking_distance = (
                (v_mps * v_mps) / (2.0 * self.red_light_decel_mps2)
                + self.stop_line_margin_m
            )

            line_dist = self._lane_stop_line_distance(light)
            if line_dist is None:
                return "proceed"
            self.stop_line_distance = line_dist

            is_red = state == carla.TrafficLightState.Red
            is_yellow = state == carla.TrafficLightState.Yellow
            if not (is_red or is_yellow):
                return "proceed"

            # Geometry: once the ego has passed the stop line the light no
            # longer applies - keep clearing the intersection.
            if line_dist <= 0.0:
                return "proceed"

            if is_red:
                if line_dist > self.required_braking_distance:
                    return "proceed"  # still approaching - braking starts at the boundary
                self.red_light_stop_active = True
                self.traffic_light_brake_command = self._light_brake(speed_kmh, line_dist)
                return "red_stop"

            # YELLOW: safe-stopping rule.
            if speed_kmh < 0.5:
                self.yellow_stop_active = True
                self.traffic_light_brake_command = self.stop_hold_brake
                return "yellow_stop"
            can_stop = line_dist >= self.required_braking_distance
            if can_stop:
                self.yellow_stop_active = True
                self.traffic_light_brake_command = self._light_brake(speed_kmh, line_dist)
                return "yellow_stop"
            return "proceed"  # too close to stop - continue through
        except Exception:
            return "proceed"

    def _light_brake(self, speed_kmh, line_dist):
        """Progressive brake that ramps up as the stop line approaches.
        A small hold-brake is kept while the ego is stationary so a red
        light keeps the car planted instead of letting it creep."""
        if speed_kmh < 0.5:
            return self.stop_hold_brake
        req = max(0.1, self.required_braking_distance)
        ratio = 1.0 - (line_dist / req)
        return max(self.stop_hold_brake, min(1.0, 0.3 + ratio * 0.7))

    def _lane_stop_line_distance(self, light):
        """Distance from the ego to ITS OWN lane's stop line, meters.
        Uses light.get_stop_waypoints() when available; each waypoint is a
        lane stop line. Prefers the waypoint on the ego's road_id+lane_id;
        falls back to the nearest waypoint that is still AHEAD of the ego.
        Returns None when no line is ahead (light not relevant / behind)."""
        try:
            ego_loc = self.vehicle.get_location()
            forward = self.vehicle.get_transform().get_forward_vector()
            ego_wp = self.world.get_map().get_waypoint(
                ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            stop_waypoints = list(light.get_stop_waypoints())
            if not stop_waypoints:
                return self._ahead_distance(ego_loc, forward, light.get_location())

            candidates = []
            for wp in stop_waypoints:
                d = self._ahead_distance(ego_loc, forward, wp.transform.location)
                if d is None:
                    continue
                same_lane = (ego_wp is None) or (
                    wp.road_id == ego_wp.road_id and wp.lane_id == ego_wp.lane_id
                )
                candidates.append((same_lane, d))
            if not candidates:
                return None
            same_lane = [d for (s, d) in candidates if s]
            pool = same_lane if same_lane else [d for (s, d) in candidates]
            return min(pool)
        except Exception:
            return None

    @staticmethod
    def _ahead_distance(ego_loc, forward, point):
        """Straight-line ground distance if `point` is in FRONT of the ego
        (positive forward dot product), else None (behind / irrelevant)."""
        try:
            dx = point.x - ego_loc.x
            dy = point.y - ego_loc.y
            if forward.x * dx + forward.y * dy <= 0.0:
                return None
            return math.sqrt(dx * dx + dy * dy)
        except Exception:
            return None

    def compute_sign_stop_scale(self):
        """Brake blend for STOP / YIELD signs (0.0 = free, 1.0 = stop).
        Traffic lights are seen via the traffic-light actor API, but stop
        and yield signs are NOT - so without this check the ego blew straight
        through them. Uses the HD map landmarks ('206' = stop, '205' =
        yield; tolerant of both code and display-name spellings)."""
        try:
            ego_wp = self.world.get_map().get_waypoint(
                self.vehicle.get_location(), project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if ego_wp is None or ego_wp.is_junction:
                return 0.0
            nearest_stop_dist = None
            for landmark in ego_wp.get_landmarks(15.0):
                lm_type = str(getattr(landmark, "type", ""))
                if ("206" in lm_type or "Stop" in lm_type
                        or "205" in lm_type or "Yield" in lm_type):
                    if nearest_stop_dist is None:
                        nearest_stop_dist = landmark.distance
                    else:
                        nearest_stop_dist = min(nearest_stop_dist, landmark.distance)
            if nearest_stop_dist is None:
                return 0.0
            scale = 1.0 - (max(0.0, nearest_stop_dist - 2.0) / 10.0)
            return max(0.0, min(1.0, scale))
        except Exception:
            return 0.0

    def decide(self, fused):
        heading_error = self.compute_heading_error()
        cross_track_error = self.compute_cross_track_error()

        # Full PID on heading error, plus cross-track lane-centering term
        self._steer_error_integral = max(
            -self.steer_integral_limit,
            min(self.steer_integral_limit, self._steer_error_integral + heading_error * self.sim_dt),
        )
        steer_error_deriv_raw = (
            0.0 if self._steer_error_previous is None
            else (heading_error - self._steer_error_previous) / self.sim_dt
        )
        self._steer_error_previous = heading_error
        if self._steer_error_deriv_f is None:
            self._steer_error_deriv_f = steer_error_deriv_raw
        else:
            self._steer_error_deriv_f = (
                self.derivative_filter_alpha * steer_error_deriv_raw
                + (1.0 - self.derivative_filter_alpha) * self._steer_error_deriv_f
            )
        steer_error_derivative = self._steer_error_deriv_f

        base_steer = (
            heading_error * self.steer_kp
            - cross_track_error * self.steer_kp_cte
            + self._steer_error_integral * self.steer_ki
            + steer_error_derivative * self.steer_kd
        )
        base_steer = max(-1.0, min(1.0, base_steer))

        # GNSS-fault steering corruption: applies only when GNSS fault is active
        gnss_bias = 0.0
        if self.mw.is_active("gps_spoof") or self.mw.is_active("gps_freeze") or self.mw.is_active("gnss_override"):
            gnss_bias = max(-0.3, min(0.3, fused["gnss_error_m"] * self.gnss_steer_error_gain))

        # IMU-fault steering corruption: applies only when an IMU fault is active
        imu_bias = 0.0
        if self.mw.is_active("imu_bias") or self.mw.is_active("imu_noise") or self.mw.is_active("imu_override"):
            imu_bias = max(-0.3, min(0.3, fused["imu_accel_x"] * self.imu_steer_bias_scale))

        # Obstacle-avoidance steer bias: smooth proportional nudge AWAY from a
        # genuinely close side obstacle. Sign: left_distance < right_distance
        # means the obstacle is on the LEFT -> steer RIGHT (negative) to get
        # clear of it. (The old sign steered INTO the closer side, which is
        # how a passing/oncoming car made the ego yank toward it and change
        # lane.) Only engages within 8 m, while the forward path is clear,
        # and ONLY when the target side actually has room (>= 12 m) - this
        # last gate stops the ego dodging straight into a wall / oncoming
        # lane when the "free" side is actually the collision side.
        avoid_bias = 0.0
        side_clearance_diff = fused["left_distance"] - fused["right_distance"]
        nearest_side = min(fused["left_distance"], fused["right_distance"])
        if (
            fused["obstacle_caution"]
            and not fused["obstacle_ahead"]
            and abs(side_clearance_diff) > 1.0
            and nearest_side < 8.0
        ):
            target_side = (
                fused["left_distance"] if side_clearance_diff > 0 else fused["right_distance"]
            )
            if target_side >= 12.0:
                nudge_direction = 1.0 if side_clearance_diff > 0 else -1.0
                avoid_bias = nudge_direction * min(
                    self.steer_avoid_gain,
                    (abs(side_clearance_diff) / 10.0) * self.steer_avoid_gain,
                )

        steer = max(-1.0, min(1.0, base_steer + gnss_bias + imu_bias + avoid_bias))

        # =========================================================
        # CLOSING SPEED & TIME-TO-COLLISION (from processed distances)
        # =========================================================
        # Sign convention (derived, NOT guessed): the front distance is
        # the middleware-processed gap to whatever is ahead. When this
        # tick's gap is SMALLER than last tick's, the ego is closing on
        # it -> positive closing speed. target_speed estimate:
        #   v_lead_front ≈ v_ego - closing          (m/s)
        # so a stopped lead ahead of a 30 km/h ego gives closing ≈ 8.3 m/s.
        front = fused["front_distance"]

        closing = 0.0
        ttc = None
        if self._prev_front is not None:
            gap_delta = self._prev_front - front
            if gap_delta > 0.05:
                closing = min(40.0, gap_delta / max(0.02, self.sim_dt))
            else:
                closing = 0.0
        self._prev_front = front
        if closing > 0.5:
            ttc = front / max(0.01, closing)

        self.last_closing_speed = closing
        self.last_ttc = ttc
        self.last_object_detected = front < self.sensor_scan_radius
        self.last_target_speed_kmh = max(0.0, fused["speed_kmh"] - closing * 3.6)

        # =========================================================
        # 1. EMERGENCY STOP (overrides ALL normal PID output)
        # =========================================================
        # Three independent triggers, any one forces brake=1.0/throttle=0:
        #   - hard distance: obstacle inside the emergency distance;
        #   - time-to-collision too short: closing fast even if still
        #     beyond the hard distance (reacts EARLY to a sudden stop /
        #     cut-in instead of only at the 5-8m band);
        #   - can-still-stop check: the front distance is center-to-center;
        #     the real bumper gap is front - 2*half_len. If the gap is
        #     closing (lead slower/then braking) and the ego cannot brake
        #     to a stop within that bumper gap, the collision would be
        #     physically inevitable - full brake NOW rather than trusting
        #     the (delayed) TTC estimate alone.
        distance_emergency = fused["obstacle_ahead"]
        ttc_emergency = (closing > 0.5) and (ttc is not None) and (ttc < self.ttc_emergency)

        v_mps = fused["speed_kmh"] / 3.6
        bumper_gap = max(0.0, front - 2.0 * self.vehicle_half_len_m)
        stop_dist_needed = (v_mps * v_mps) / (2.0 * self.emergency_decel_mps2)
        can_stop_in_gap = bumper_gap >= (stop_dist_needed + 1.5)
        # Lead-speed-aware runaway: a lead that STOPS hard drops its speed
        # estimate below 10 km/h. If the ego can no longer brake to a halt
        # inside the bumper gap, collision is physically inevitable - full
        # brake NOW, instead of waiting on TTC (which reads ~0/0 exactly
        # when the lead is already at a standstill).
        lead_speed_kmh = max(0.0, fused["speed_kmh"] - closing * 3.6)
        lead_stopped = lead_speed_kmh < 10.0
        runaway = (closing > 0.5 or lead_stopped) and (not can_stop_in_gap) and (v_mps > 1.0)
        # A cut-in can drop front from ~30 m to ~6 m in ONE tick; closing
        # stays 0 (the delta is negative) so TTC never fires - but a bumper
        # gap under 2 m is already unavoidable -> full brake regardless.
        bumper_emergency = bumper_gap < 2.0

        emergency_active = bool(distance_emergency or ttc_emergency or runaway or bumper_emergency)

        self.last_bumper_gap = bumper_gap
        self.last_stop_dist_needed = stop_dist_needed
        self.last_lead_speed_kmh = lead_speed_kmh

        if emergency_active:
            self.last_effective_cruise = 0.0
            self.last_speed_error = -fused["speed_kmh"]
            self.last_command = 0.0
            self.last_mode = "emergency_brake"
            self.last_emergency = True
            if not self._in_emergency:
                print(
                    f"[EMERGENCY BRAKE] dist={distance_emergency} ttc={ttc_emergency} "
                    f"runaway={runaway} bumper={bumper_emergency} "
                    f"front={front:.1f}m gap={bumper_gap:.1f}m "
                    f"v={fused['speed_kmh']:.1f}km/h target_stop_dist={stop_dist_needed:.1f}m"
                )
                self._in_emergency = True
            control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=steer, hand_brake=False)
            return control, "emergency_brake"

        self.last_emergency = False
        self._in_emergency = False

        # =========================================================
        # 2. TRAFFIC-LIGHT HARD STOP (above the normal PID)
        # =========================================================
        # Red/Yellow are HARD stops: throttle 0 + a direct brake command,
        # NOT a cruise-speed blend. (GOAL 1 - the old gentle blend let the
        # PID fight the brake and the ego rolled over the stop line.)
        # compute_traffic_light_stop() only fires for a stop line belonging
        # to the EGO'S OWN lane, and it holds the car stopped until green.
        light_mode = self.compute_traffic_light_stop(fused["speed_kmh"])
        traffic_stop = light_mode in ("red_stop", "yellow_stop")

        # Stop/YIELD signs stay as a rule blend (no stop-waypoint API for
        # them; HD-map landmarks are the only source).
        rule_stop_scale = self.compute_sign_stop_scale()
        self.last_rule_stop_scale = rule_stop_scale

        # One-shot [RED LIGHT STOP] / [YELLOW LIGHT STOP] event prints
        if traffic_stop and not self._traffic_stop_was_active:
            tag = "RED" if light_mode == "red_stop" else "YELLOW"
            print(
                f"[{tag} LIGHT STOP] line={self.stop_line_distance:.1f}m "
                f"required={self.required_braking_distance:.1f}m "
                f"brake={self.traffic_light_brake_command:.2f} "
                f"v={fused['speed_kmh']:.1f}km/h target_stop_dist=0.0"
            )
            self._traffic_stop_was_active = True
        elif not traffic_stop:
            self._traffic_stop_was_active = False

        if traffic_stop:
            self.last_effective_cruise = 0.0
            self.last_speed_error = -fused["speed_kmh"]
            self.last_command = 0.0
            self.last_mode = (
                "red_light_stop" if light_mode == "red_stop" else "yellow_light_stop"
            )
            control = carla.VehicleControl(
                throttle=0.0,
                brake=self.traffic_light_brake_command,
                steer=steer,
                hand_brake=False,
            )
            return control, self.last_mode

        # =========================================================
        # 3. OBSTACLE CAUTION / NORMAL PID (longitudinal)
        # =========================================================

        # Closing-speed-aware caution. When the gap is closing (TTC below
        # the caution threshold, or already inside the static band) the
        # speed-PID target is blended DOWN toward 0. The band is widened
        # by the closing-speed look-ahead (reaction_time), so approaching
        # traffic is slowed for EARLY and smoothly instead of only inside
        # the static band. (The old code - static 5-8m band only - did
        # nothing until the last moment, which is the root cause of the
        # tail-end collisions.)
        rule_caution = fused["obstacle_caution"] or (
            (closing > 0.5) and (ttc is not None) and (ttc < self.ttc_caution)
        )
        obstacle_scale = 0.0
        if rule_caution:
            vfront = max(0.1, front - closing * self.reaction_time)
            obstacle_scale = 1.0 - (
                (vfront - self.emergency_brake_distance) / max(0.1, self.caution_zone_m)
            )
            obstacle_scale = max(0.0, min(1.0, obstacle_scale))
        self.last_obstacle_scale = obstacle_scale

        # One-shot [OBSTACLE CAUTION] event print on caution entry
        if rule_caution and not self._caution_was_active:
            print(
                f"[OBSTACLE CAUTION] front={front:.1f}m ttc="
                f"{ttc if ttc is not None else 0.0:.1f}s "
                f"closing={closing:.1f}m/s obstacle_scale={obstacle_scale:.2f}"
            )
        self._caution_was_active = bool(rule_caution)

        speed_scaler = max(1.0 - obstacle_scale, 1.0 - rule_stop_scale)
        effective_cruise = self.cruise_speed_kmh * speed_scaler

        speed_error = effective_cruise - fused["speed_kmh"]

        self._speed_error_integral = max(
            -self.speed_integral_limit,
            min(self.speed_integral_limit, self._speed_error_integral + speed_error * self.sim_dt),
        )
        speed_error_deriv_raw = (
            0.0 if self._speed_error_previous is None
            else (speed_error - self._speed_error_previous) / self.sim_dt
        )
        self._speed_error_previous = speed_error
        if self._speed_error_deriv_f is None:
            self._speed_error_deriv_f = speed_error_deriv_raw
        else:
            self._speed_error_deriv_f = (
                self.derivative_filter_alpha * speed_error_deriv_raw
                + (1.0 - self.derivative_filter_alpha) * self._speed_error_deriv_f
            )
        speed_error_derivative = max(-40.0, min(40.0, self._speed_error_deriv_f))

        command = (
            speed_error * self.speed_kp
            + self._speed_error_integral * self.speed_ki
            + speed_error_derivative * self.speed_kd
        )

        # Standstill launch only when the path ahead is clear and the signal
        # is green; near an obstacle or a red light let the blended target
        # hold the car instead.
        if fused["speed_kmh"] < 5.0 and obstacle_scale < 0.5 and rule_stop_scale < 0.5:
            throttle = max(0.50, min(1.0, command))
            brake = 0.0
        elif command >= 0:
            throttle = max(0.0, min(1.0, command))
            brake = 0.0
        else:
            throttle = 0.0
            # Coasting deadband: only apply friction brake if overspeeding
            # the blended target (which is reduced near obstacles).
            if speed_error < -3.0:
                brake = max(0.0, min(0.6, -command * 0.5))
            else:
                brake = 0.0

        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer, hand_brake=False)

        self.last_effective_cruise = effective_cruise
        self.last_speed_error = speed_error
        self.last_command = command
        self.last_mode = "sensor_driving"
        return control, "sensor_driving"
