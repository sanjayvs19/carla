import carla
import pygame
import numpy as np
import random
import time
import math
import threading
from collections import deque
from pynput import keyboard


# =========================================================
# SETTINGS
# =========================================================

HOST = "127.0.0.1"
PORT = 2000
TM_PORT = 8000

MAX_TRAFFIC = 100

MAX_PEDESTRIANS = 15

POTHOLE_COUNT = 15

RASH_DRIVER_PERCENTAGE = 0.35   # fraction of traffic vehicles that speed / run lights / tailgate

PEDESTRIAN_CROSS_FACTOR = 0.07   # higher = more pedestrians jaywalk instead of using crosswalks

# --- Ego collision-avoidance braking ---
# --- Emergency-brake distance must stay BELOW the Traffic Manager's own
# following gap (6.0m, set via distance_to_leading_vehicle below), or the
# override fires during completely normal car-following and the ego brakes
# to a stop constantly - which is why the speed graph was pinned at 0 the
# whole session. 5.0m leaves genuine emergencies (cut-ins, jaywalkers) as
# the only thing that trips it.
EMERGENCY_BRAKE_DISTANCE = 5.0   # meters - if something is closer than this, ahead, brake hard
EMERGENCY_BRAKE_CONE_DEGREES = 35.0  # how wide "in front of the car" counts

# --- Automatic reverse (backs up on its own if genuinely stuck) ---
STUCK_TICKS_THRESHOLD = 60     # ~3 seconds at 20 ticks/sec of being blocked + stopped
REVERSE_DURATION_TICKS = 40    # ~2 seconds of backing up before trying forward again

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
# NOTE: these are exponential-smoothing factors applied once per tick
# (20 ticks/sec). At 0.01 the camera only closes 1% of the distance to
# its target every tick, which has a ~5s time constant - the vehicle
# simply outruns it and the chase cam drifts further and further
# behind (that's the "lag" seen in the recording, where the Tesla
# never actually appears in frame). 0.15-0.2 keeps the motion smooth
# while actually staying locked onto the car.

# ------------------------------------------------------------
# WINDOW / LAYOUT BUDGET
# The HUD is split across THREE independent OS windows - Data,
# Cameras, and Analytics - each sized for its own content instead
# of all three fighting over one shrinking window. Everything
# below is derived from these numbers so nothing inside a given
# window can overlap - change these first if it still doesn't fit.
# ------------------------------------------------------------

# --- Data window (title + both text columns) ---
TEXT_COL_A_X = 30
TEXT_COL_B_X = 400
TEXT_COL_WIDTH = 350
DATA_WIN_WIDTH = 820
DATA_WIN_HEIGHT = 900

# --- Camera window (4 directional feeds + DVS + pedal bars) ---
# "a little bigger" per request -> nearly doubled from the original
# 190x140 thumbnails.
SENSOR_CAM_WIDTH = 320
SENSOR_CAM_HEIGHT = 240
CAMERA_GRID_GAP = 20
CAMERA_MARGIN = 30

# --- Sensor chart / analytics window ---
HISTORY_LENGTH = 200   # rolling samples kept for the line charts (~10s @ 20fps)

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


    except AttributeError:

        pass


keyboard_listener = keyboard.Listener(
    on_press=on_press
)

keyboard_listener.start()


# =========================================================
# CONNECT TO CARLA
# =========================================================

client = carla.Client(
    HOST,
    PORT
)

client.set_timeout(15.0)

world = client.get_world()

blueprints = world.get_blueprint_library()

print("Connected to CARLA")

print(
    "Map:",
    world.get_map().name
)


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

traffic_manager = client.get_trafficmanager(
    TM_PORT
)

traffic_manager.set_synchronous_mode(
    True
)

traffic_manager.set_hybrid_physics_mode(
    True
)

traffic_manager.set_hybrid_physics_radius(
    50.0
)

traffic_manager.set_global_distance_to_leading_vehicle(
    6.0
)

traffic_manager.global_percentage_speed_difference(
    0.0
)

traffic_manager.set_random_device_seed(
    10
)


# =========================================================
# SPAWN POINTS
# =========================================================

spawn_points = (
    world.get_map().get_spawn_points()
)

if not spawn_points:

    print("No spawn points found.")

    keyboard_listener.stop()

    raise SystemExit


# =========================================================
# SPAWN TESLA
# =========================================================

tesla_bp = blueprints.find(
    "vehicle.tesla.model3"
)

# Hybrid physics mode (enabled above) only keeps FULL physics running for
# vehicles tagged role_name="hero", and for anything within
# hybrid_physics_radius of one. With no vehicle tagged "hero" at all, the
# Traffic Manager's own docs say it disables physics for EVERY vehicle,
# including the ego - so the car gets moved by teleportation instead of
# real simulated physics. That's why the IMU accelerometer and the
# throttle/brake pedal readings were never accurate: there's no real
# physics being computed to read them from. Tagging the Tesla as "hero"
# guarantees it (and anything near it) always runs full physics.
if tesla_bp.has_attribute("role_name"):
    tesla_bp.set_attribute("role_name", "hero")

vehicle = None

tesla_spawn = None

random.seed(10)

random.shuffle(
    spawn_points
)


for sp in spawn_points:

    vehicle = world.try_spawn_actor(
        tesla_bp,
        sp
    )

    if vehicle is not None:

        tesla_spawn = sp

        break


if vehicle is None:

    print(
        "Failed to spawn Tesla."
    )

    keyboard_listener.stop()

    raise SystemExit

# Belt-and-braces alongside the "hero" tag above: make sure the ego
# never gets left in the physics-disabled / teleport-only state that
# hybrid physics mode can put vehicles into.
vehicle.set_simulate_physics(True)


print(
    "Tesla Model 3 spawned successfully."
)


# =========================================================
# AUTOPILOT
# =========================================================

vehicle.set_autopilot(
    True,
    traffic_manager.get_port()
)

traffic_manager.auto_lane_change(
    vehicle,
    True
)

traffic_manager.distance_to_leading_vehicle(
    vehicle,
    6.0
)

traffic_manager.vehicle_percentage_speed_difference(
    vehicle,
    0.3
)

traffic_manager.ignore_vehicles_percentage(
    vehicle,
    -10.0
)

traffic_manager.ignore_walkers_percentage(
    vehicle,
    100.0
)

traffic_manager.ignore_lights_percentage(
    vehicle,
    100.0
)

traffic_manager.ignore_signs_percentage(
    vehicle,
    100.0
)


# =========================================================
# TRAFFIC
# =========================================================

traffic_vehicles = []

traffic_blueprints = (
    blueprints.filter("vehicle.*")
)


traffic_blueprints = [

    bp

    for bp in traffic_blueprints

    if "ambulance" not in bp.id

    and "firetruck" not in bp.id

    and "police" not in bp.id
]


traffic_spawn_points = (
    spawn_points.copy()
)

random.shuffle(
    traffic_spawn_points
)


for sp in traffic_spawn_points:

    if len(traffic_vehicles) >= MAX_TRAFFIC:

        break


    distance = (
        sp.location.distance(
            tesla_spawn.location
        )
    )


    if distance < 30:

        continue


    bp = random.choice(
        traffic_blueprints
    )


    if bp.has_attribute("color"):

        colors = (
            bp.get_attribute(
                "color"
            ).recommended_values
        )

        if colors:

            bp.set_attribute(
                "color",
                random.choice(colors)
            )


    npc = world.try_spawn_actor(
        bp,
        sp
    )


    if npc is not None:

        npc.set_autopilot(
            True,
            traffic_manager.get_port()
        )


        is_rash_driver = (
            random.random() < RASH_DRIVER_PERCENTAGE
        )


        if is_rash_driver:

            # Rare and mild - occasionally speeds a bit and bends a
            # rule, without being a constant hazard.

            traffic_manager.auto_lane_change(
                npc,
                True
            )

            traffic_manager.distance_to_leading_vehicle(
                npc,
                1.0
            )

            traffic_manager.vehicle_percentage_speed_difference(
                npc,
                random.uniform(-60.0, -25.0)   # negative = faster than speed limit
            )

            traffic_manager.ignore_lights_percentage(
                npc,
                80.0
            )

            traffic_manager.ignore_signs_percentage(
                npc,
                80.0
            )

            traffic_manager.random_left_lanechange_percentage(
                npc,
                60.0
            )

            traffic_manager.random_right_lanechange_percentage(
                npc,
                60.0
            )

        else:

            traffic_manager.auto_lane_change(
                npc,
                True
            )


            traffic_manager.distance_to_leading_vehicle(
                npc,
                6.0
            )


            traffic_manager.vehicle_percentage_speed_difference(
                npc,
                random.uniform(-5.0, 5.0)
            )


        traffic_vehicles.append(
            npc
        )


print(
    f"{len(traffic_vehicles)} traffic vehicles spawned "
    f"({int(RASH_DRIVER_PERCENTAGE * 100)}% rash/rule-violating)."
)


# =========================================================
# PEDESTRIANS (with jaywalking)
# =========================================================

# Higher cross factor = more pedestrians cross roads outside of
# crosswalks instead of only at marked crossings (jaywalking).
world.set_pedestrians_cross_factor(
    PEDESTRIAN_CROSS_FACTOR
)

walker_blueprints = (
    blueprints.filter("walker.pedestrian.*")
)

walker_controller_bp = blueprints.find(
    "controller.ai.walker"
)

pedestrians = []
pedestrian_controllers = []

for _ in range(MAX_PEDESTRIANS):

    nav_location = (
        world.get_random_location_from_navigation()
    )

    if nav_location is None:
        continue

    bp = random.choice(
        walker_blueprints
    )

    walker = world.try_spawn_actor(
        bp,
        carla.Transform(nav_location)
    )

    if walker is None:
        continue

    controller = world.spawn_actor(
        walker_controller_bp,
        carla.Transform(),
        attach_to=walker
    )

    controller.start()

    controller.go_to_location(
        world.get_random_location_from_navigation()
    )

    # Wider speed range so some pedestrians hurry across
    # traffic instead of calmly walking - more realistic mix.
    controller.set_max_speed(
        random.uniform(1.2, 2.5)
    )

    pedestrians.append(walker)
    pedestrian_controllers.append(controller)


print(
    f"{len(pedestrians)} pedestrians spawned "
    f"(cross factor {PEDESTRIAN_CROSS_FACTOR} - some will jaywalk)."
)


# =========================================================
# POTHOLES (visual hazard markers, non-collidable)
# =========================================================

# A solid mesh prop placed on the road would block traffic lanes
# and cause the same gridlock issues fixed earlier, so potholes
# are drawn as persistent flat red markers directly on the road
# surface - visible in the CARLA window and logged with real
# world coordinates, but with no collision to disrupt driving.

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
        life_time=0.0   # 0 = persists for the whole session
    )

    world.debug.draw_string(
        pothole_center + carla.Location(z=0.3),
        "POTHOLE",
        draw_shadow=False,
        color=carla.Color(255, 60, 0),
        life_time=0.0
    )

    pothole_locations.append(pothole_center)


print(
    f"{len(pothole_locations)} pothole markers placed on the road network."
)


# =========================================================
# SPECTATOR
# =========================================================

spectator = world.get_spectator()


# =========================================================
# HAZARD SCAN (distance to nearest object, all sides)
# =========================================================

def scan_surroundings():
    """Returns nearest-object distance (meters) in front / back / left /
    right of the ego car, by checking every traffic vehicle and
    pedestrian against the car's own facing direction. Used both for
    the HUD readout and the emergency braking check below."""

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

        distance = math.sqrt(
            delta.x ** 2 + delta.y ** 2
        )

        if distance < 0.5 or distance > SENSOR_SCAN_RADIUS:
            continue

        forward_component = (
            delta.x * forward.x + delta.y * forward.y
        )

        right_component = (
            delta.x * right.x + delta.y * right.y
        )

        if abs(forward_component) >= abs(right_component):

            if forward_component >= 0:
                key = "front"
            else:
                key = "back"

        else:

            if right_component >= 0:
                key = "right"
            else:
                key = "left"

        if distance < nearest[key]:
            nearest[key] = distance

    return nearest


def front_hazard_distance(nearest):
    """Narrower forward cone check specifically for emergency braking,
    so the car only brakes for things genuinely in its path."""

    transform = vehicle.get_transform()

    origin = transform.location

    forward = transform.get_forward_vector()

    closest = SENSOR_SCAN_RADIUS

    candidates = list(traffic_vehicles) + list(pedestrians)

    for other in candidates:

        if other is None or not other.is_alive:
            continue

        other_location = other.get_location()

        delta = other_location - origin

        distance = math.sqrt(
            delta.x ** 2 + delta.y ** 2
        )

        if distance < 0.5 or distance > EMERGENCY_BRAKE_DISTANCE + 3.0:
            continue

        forward_component = (
            delta.x * forward.x + delta.y * forward.y
        )

        if forward_component <= 0:
            continue

        angle = math.degrees(
            math.acos(
                max(-1.0, min(1.0, forward_component / distance))
            )
        )

        if angle <= EMERGENCY_BRAKE_CONE_DEGREES and distance < closest:
            closest = distance

    return closest


# =========================================================
# DIRECTIONAL CAMERAS (front / back / left / right feeds)
# =========================================================

camera_frames = {}

camera_bp = blueprints.find(
    "sensor.camera.rgb"
)

camera_bp.set_attribute(
    "image_size_x", str(SENSOR_CAM_WIDTH)
)

camera_bp.set_attribute(
    "image_size_y", str(SENSOR_CAM_HEIGHT)
)

# Only capture ~6-7 times per second instead of every single
# simulation tick (20/sec) - this is the main lag fix. The HUD
# thumbnails don't need full tick-rate video to be useful.
camera_bp.set_attribute(
    "sensor_tick", "0.15"
)

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

    cam = world.spawn_actor(
        camera_bp,
        cam_transform,
        attach_to=vehicle
    )

    def make_callback(name):

        def callback(image):
            array = np.frombuffer(
                image.raw_data, dtype=np.uint8
            )
            array = array.reshape(
                (image.height, image.width, 4)
            )
            camera_frames[name] = array

        return callback

    cam.listen(make_callback(cam_name))

    directional_cameras.append(cam)


print(
    f"{len(directional_cameras)} directional cameras attached "
    "(front, back, left, right)."
)


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
gnss_sensor = world.spawn_actor(
    gnss_bp, carla.Transform(), attach_to=vehicle
)


def on_gnss(data):
    sensor_readings["gnss"]["lat"] = data.latitude
    sensor_readings["gnss"]["lon"] = data.longitude
    sensor_readings["gnss"]["alt"] = data.altitude


gnss_sensor.listen(on_gnss)


# --- IMU (accelerometer + gyroscope + compass) ---
imu_bp = blueprints.find("sensor.other.imu")
imu_sensor = world.spawn_actor(
    imu_bp, carla.Transform(), attach_to=vehicle
)


def on_imu(data):
    sensor_readings["imu"]["accel_x"] = data.accelerometer.x
    sensor_readings["imu"]["accel_y"] = data.accelerometer.y
    sensor_readings["imu"]["accel_z"] = data.accelerometer.z
    sensor_readings["imu"]["gyro_x"] = data.gyroscope.x
    sensor_readings["imu"]["gyro_y"] = data.gyroscope.y
    sensor_readings["imu"]["gyro_z"] = data.gyroscope.z
    sensor_readings["imu"]["compass"] = math.degrees(data.compass)


imu_sensor.listen(on_imu)


# --- Collision sensor ---
collision_bp = blueprints.find("sensor.other.collision")
collision_sensor = world.spawn_actor(
    collision_bp, carla.Transform(), attach_to=vehicle
)


def on_collision(event):
    sensor_readings["collision"] = event.other_actor.type_id


collision_sensor.listen(on_collision)


# --- Lane invasion sensor ---
lane_bp = blueprints.find("sensor.other.lane_invasion")
lane_sensor = world.spawn_actor(
    lane_bp, carla.Transform(), attach_to=vehicle
)


def on_lane_invasion(event):
    crossed_types = [
        str(marking.type) for marking in event.crossed_lane_markings
    ]
    sensor_readings["lane_invasion"] = (
        ", ".join(crossed_types) if crossed_types else "None"
    )


lane_sensor.listen(on_lane_invasion)


# --- LiDAR ---
lidar_bp = blueprints.find("sensor.lidar.ray_cast")
lidar_bp.set_attribute("channels", "32")
lidar_bp.set_attribute("range", "40")
lidar_bp.set_attribute("points_per_second", "56000")
lidar_bp.set_attribute("rotation_frequency", "20")
lidar_bp.set_attribute("sensor_tick", "0.1")   # throttled for performance

lidar_sensor = world.spawn_actor(
    lidar_bp, carla.Transform(carla.Location(z=2.2)), attach_to=vehicle
)


def on_lidar(data):
    points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
    # Downsample for a readable/affordable scatter plot - a full sweep
    # can be tens of thousands of points, we only need a few hundred
    # to show the shape of what's around the car.
    if len(points) > 600:
        idx = np.random.choice(len(points), 600, replace=False)
        points = points[idx]
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
    sensor_readings["obstacle"] = {
        "actor": other_name,
        "distance": event.distance,
        "time": time.time(),
    }


obstacle_sensor.listen(on_obstacle)


# --- DVS (event) camera ---
dvs_bp = blueprints.find("sensor.camera.dvs")
dvs_bp.set_attribute("image_size_x", str(SENSOR_CAM_WIDTH))
dvs_bp.set_attribute("image_size_y", str(SENSOR_CAM_HEIGHT))
dvs_bp.set_attribute("sensor_tick", "0.15")

dvs_sensor = world.spawn_actor(
    dvs_bp, camera_rig_transforms["front"], attach_to=vehicle
)

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

smooth_position = (
    vehicle.get_location()
)

smooth_yaw = (
    vehicle.get_transform()
    .rotation.yaw
)

smooth_pitch = -18.0

orbit_angle = 0.0


# =========================================================
# CAMERA SMOOTH ANGLE
# =========================================================

def smooth_angle(
    current,
    target,
    factor
):

    difference = (

        (target - current + 180.0)

        % 360.0

    ) - 180.0


    return (
        current
        + difference * factor
    )


# =========================================================
# CAMERA UPDATE
# =========================================================

def update_camera():

    global smooth_position

    global smooth_yaw

    global smooth_pitch

    global orbit_angle


    transform = (
        vehicle.get_transform()
    )


    vehicle_location = (
        transform.location
    )


    vehicle_yaw = (
        transform.rotation.yaw
    )


    with camera_lock:

        current_mode = camera_mode


    # =====================================================
    # CHASE CAMERA
    # =====================================================

    if current_mode == "chase":

        forward = (
            transform.get_forward_vector()
        )


        target_position = (

            vehicle_location

            - forward * CAMERA_DISTANCE

            + carla.Location(
                z=CAMERA_HEIGHT
            )
        )


        target_yaw = vehicle_yaw

        target_pitch = -18.0


    # =====================================================
    # FRONT CAMERA
    # =====================================================

    elif current_mode == "front":

        forward = (
            transform.get_forward_vector()
        )


        target_position = (

            vehicle_location

            + forward * 7.0

            + carla.Location(
                z=2.5
            )
        )


        target_yaw = vehicle_yaw

        target_pitch = -5.0


    # =====================================================
    # BACK CAMERA
    # =====================================================

    elif current_mode == "back":

        forward = (
            transform.get_forward_vector()
        )


        target_position = (

            vehicle_location

            - forward * 8.0

            + carla.Location(
                z=3.0
            )
        )


        target_yaw = (
            vehicle_yaw + 180.0
        )


        target_pitch = -10.0


    # =====================================================
    # TOP CAMERA
    # =====================================================

    elif current_mode == "top":

        target_position = (

            vehicle_location

            + carla.Location(
                z=12.0
            )
        )


        target_yaw = vehicle_yaw

        target_pitch = -90.0


    # =====================================================
    # LOW CAMERA
    # =====================================================

    elif current_mode == "bottom":

        forward = (
            transform.get_forward_vector()
        )


        target_position = (

            vehicle_location

            + forward * 2.0

            + carla.Location(
                z=1.2
            )
        )


        target_yaw = vehicle_yaw

        target_pitch = -2.0


    # =====================================================
    # 360 CAMERA
    # =====================================================

    elif current_mode == "360":

        orbit_angle += 0.8


        radians = math.radians(
            orbit_angle
        )


        radius = 12.0


        x = (
            math.cos(radians)
            * radius
        )


        y = (
            math.sin(radians)
            * radius
        )


        target_position = (

            vehicle_location

            + carla.Location(
                x=x,
                y=y,
                z=6.0
            )
        )


        target_yaw = (

            math.degrees(
                radians
            )
            + 180.0
        )


        target_pitch = -15.0


    else:

        target_position = (

            vehicle_location

            + carla.Location(
                z=5.0
            )
        )


        target_yaw = vehicle_yaw

        target_pitch = -15.0


    # =====================================================
    # SMOOTH POSITION
    # =====================================================

    smooth_position = carla.Location(

        x=smooth_position.x

        + (

            target_position.x
            - smooth_position.x

        ) * POSITION_SMOOTH,


        y=smooth_position.y

        + (

            target_position.y
            - smooth_position.y

        ) * POSITION_SMOOTH,


        z=smooth_position.z

        + (

            target_position.z
            - smooth_position.z

        ) * POSITION_SMOOTH
    )


    # =====================================================
    # SMOOTH ROTATION
    # =====================================================

    smooth_yaw = smooth_angle(

        smooth_yaw,

        target_yaw,

        ROTATION_SMOOTH
    )


    smooth_pitch += (

        target_pitch
        - smooth_pitch

    ) * ROTATION_SMOOTH


    # =====================================================
    # APPLY CAMERA
    # =====================================================

    spectator.set_transform(

        carla.Transform(

            smooth_position,

            carla.Rotation(

                pitch=smooth_pitch,

                yaw=smooth_yaw,

                roll=0.0
            )
        )
    )


# =========================================================
# PYGAME - THREE SEPARATE WINDOWS
# =========================================================
# Instead of one crowded window, the HUD is split into three
# independent OS windows (Data / Cameras / Analytics) that the
# user can drag apart, move to separate monitors, resize freely,
# or close independently. Each window gets its own plain Surface
# to draw on exactly like before (pygame.draw / blit all still
# work unchanged) - the only new part is converting that Surface
# to a Texture and presenting it to its own Renderer each frame.

pygame.init()

from pygame._sdl2 import video as sdl2_video

display_info = pygame.display.Info()
SCREEN_W = display_info.current_w
SCREEN_H = display_info.current_h

# --- Camera window sizing ---
# SENSOR_CAM_WIDTH/HEIGHT (320x240) is the actual CARLA sensor
# capture resolution - that's already baked into the blueprints and
# can't change here. CAM_TILE_WIDTH/HEIGHT is the on-screen size of
# each tile in the grid, which defaults to a 1:1 match but can be
# scaled down (with a smoothscale at blit time) if the window
# wouldn't otherwise fit the user's actual screen.
CAM_TILE_WIDTH = SENSOR_CAM_WIDTH
CAM_TILE_HEIGHT = SENSOR_CAM_HEIGHT

CAMERA_WIN_WIDTH = (
    CAMERA_MARGIN + 2 * CAM_TILE_WIDTH + CAMERA_GRID_GAP
    + CAMERA_GRID_GAP + CAM_TILE_WIDTH + CAMERA_MARGIN
)
CAMERA_WIN_HEIGHT = (
    70 + 2 * CAM_TILE_HEIGHT + CAMERA_GRID_GAP + CAMERA_MARGIN
)

max_camera_w = SCREEN_W - 100
max_camera_h = SCREEN_H - 150

if CAMERA_WIN_WIDTH > max_camera_w or CAMERA_WIN_HEIGHT > max_camera_h:

    cam_scale = min(
        max_camera_w / CAMERA_WIN_WIDTH, max_camera_h / CAMERA_WIN_HEIGHT
    )

    CAM_TILE_WIDTH = int(CAM_TILE_WIDTH * cam_scale)
    CAM_TILE_HEIGHT = int(CAM_TILE_HEIGHT * cam_scale)

    CAMERA_WIN_WIDTH = (
        CAMERA_MARGIN + 2 * CAM_TILE_WIDTH + CAMERA_GRID_GAP
        + CAMERA_GRID_GAP + CAM_TILE_WIDTH + CAMERA_MARGIN
    )
    CAMERA_WIN_HEIGHT = (
        70 + 2 * CAM_TILE_HEIGHT + CAMERA_GRID_GAP + CAMERA_MARGIN
    )

# --- Analytics window sizing ---
# 2 columns x 3 stacked charts each - scale the chart panels down
# (same technique) if the ideal size wouldn't fit the screen.
CHART_WIN_WIDTH = CHART_MARGIN + 2 * CHART_WIDTH + CHART_GAP + CHART_MARGIN
CHART_WIN_HEIGHT = 60 + 3 * CHART_HEIGHT + 2 * CHART_GAP + CHART_MARGIN

max_chart_w = SCREEN_W - 100
max_chart_h = SCREEN_H - 150

if CHART_WIN_WIDTH > max_chart_w or CHART_WIN_HEIGHT > max_chart_h:

    chart_scale = min(
        max_chart_w / CHART_WIN_WIDTH, max_chart_h / CHART_WIN_HEIGHT
    )

    CHART_WIDTH = int(CHART_WIDTH * chart_scale)
    CHART_HEIGHT = int(CHART_HEIGHT * chart_scale)
    CHART_GAP = int(CHART_GAP * chart_scale)
    CHART_COL1_X = CHART_MARGIN
    CHART_COL2_X = CHART_MARGIN + CHART_WIDTH + CHART_GAP

    CHART_WIN_WIDTH = CHART_MARGIN + 2 * CHART_WIDTH + CHART_GAP + CHART_MARGIN
    CHART_WIN_HEIGHT = 60 + 3 * CHART_HEIGHT + 2 * CHART_GAP + CHART_MARGIN

# --- Data window sizing ---
DATA_WIN_HEIGHT = min(DATA_WIN_HEIGHT, SCREEN_H - 150)

# Cascade the three windows on launch so they don't stack exactly on
# top of each other - the user can then drag each wherever they like
# (including onto separate monitors).
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
    """Push a plain Surface's pixels to its window for this frame."""
    texture = sdl2_video.Texture.from_surface(renderer, surface)
    renderer.clear()
    texture.draw()
    renderer.present()


font_title = pygame.font.Font(
    None,
    32
)


font_data = pygame.font.Font(
    None,
    23
)


font_axis = pygame.font.Font(
    None,
    16
)


clock = pygame.time.Clock()


# =========================================================
# CHART DRAWING HELPERS
# =========================================================

def draw_chart_frame(surface, rect, title,
                      y_min=None, y_max=None, y_unit="",
                      x_min=None, x_max=None, x_unit="",
                      x_is_time=False):
    """Draws the shared dark box + title + inner plot area used by
    every chart, with numeric axis tick labels, and returns the
    inner plot rect to draw series/points into.

    y_min/y_max: if given, draws 5 numeric ticks up the left edge.
    x_is_time:   if True, labels the x-axis "-10s" ... "now"
                 (used by the rolling history line charts).
    x_min/x_max: if given (and x_is_time is False), draws 5 numeric
                 ticks along the bottom edge - used by scatter charts
                 like LiDAR (meters) and Radar (degrees).
    """

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

    # Horizontal gridlines
    for i in range(1, 4):
        gy = plot_rect.y + plot_rect.height * i // 4
        pygame.draw.line(
            surface, (42, 42, 42),
            (plot_rect.x, gy), (plot_rect.right, gy), 1
        )

    # Vertical gridlines (only meaningful for value-scaled x-axes)
    if x_min is not None and not x_is_time:
        for i in range(1, 4):
            gx = plot_rect.x + plot_rect.width * i // 4
            pygame.draw.line(
                surface, (42, 42, 42),
                (gx, plot_rect.y), (gx, plot_rect.bottom), 1
            )

    # Y-axis numeric ticks
    if y_min is not None and y_max is not None:
        for i in range(5):
            frac = i / 4
            gy = plot_rect.bottom - frac * plot_rect.height
            val = y_min + frac * (y_max - y_min)
            label = font_axis.render(f"{val:.0f}{y_unit}", True, (140, 140, 140))
            surface.blit(
                label,
                (plot_rect.x - label.get_width() - 5, gy - label.get_height() // 2)
            )

    # X-axis labels
    if x_is_time:
        left_label = font_axis.render("-10s", True, (140, 140, 140))
        right_label = font_axis.render("now", True, (140, 140, 140))
        surface.blit(left_label, (plot_rect.x, plot_rect.bottom + 4))
        surface.blit(
            right_label,
            (plot_rect.right - right_label.get_width(), plot_rect.bottom + 4)
        )
    elif x_min is not None and x_max is not None:
        for i in range(5):
            frac = i / 4
            gx = plot_rect.x + frac * plot_rect.width
            val = x_min + frac * (x_max - x_min)
            label = font_axis.render(f"{val:.0f}{x_unit}", True, (140, 140, 140))
            surface.blit(
                label,
                (gx - label.get_width() // 2, plot_rect.bottom + 4)
            )

    return plot_rect


def draw_line_series(surface, plot_rect, data, min_val, max_val, color):
    """Plots one rolling history series as a line inside plot_rect."""

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
    """Full line-chart panel: frame + series + current value readout.

    auto_scale zooms the y-axis in around the data actually seen
    recently (clamped to [min_val, max_val]) instead of always using
    the full fixed range. Fixed full-range axes (e.g. 0-140 km/h)
    squash normal city-driving values into a thin band near the
    bottom of the chart, which reads as "no variation" even though
    the underlying values are changing plenty."""

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
        surface, rect, title,
        y_min=plot_min, y_max=plot_max, y_unit=unit,
        x_is_time=True
    )

    draw_line_series(surface, plot_rect, data, plot_min, plot_max, color)

    current_val = data[-1] if data else 0.0

    value_label = font_data.render(f"{current_val:.1f}{unit}", True, color)
    surface.blit(value_label, (rect.x + 12, rect.bottom - 20))


def draw_bar_chart(surface, rect, title, categories, values, max_val, warn_val, danger_val):
    """Bar chart used for the sensor distance readout (front/back/left/right),
    colour-coded the same way as the camera thumbnail borders."""

    plot_rect = draw_chart_frame(
        surface, rect, title,
        y_min=0.0, y_max=max_val, y_unit="m"
    )

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
        surface.blit(
            value_label,
            (bx + bar_width // 2 - value_label.get_width() // 2, label_y)
        )

        name_label = font_data.render(category, True, (220, 220, 220))
        surface.blit(
            name_label,
            (bx + bar_width // 2 - name_label.get_width() // 2, plot_rect.bottom + 4)
        )


def draw_dual_line_chart(surface, rect, title, data_a, label_a, color_a,
                          data_b, label_b, color_b, min_val, max_val, unit="", auto_scale=True):
    """Line chart with two overlaid series (used for throttle vs brake,
    and LiDAR left vs right)."""

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
        surface, rect, title,
        y_min=plot_min, y_max=plot_max, y_unit=unit,
        x_is_time=True
    )

    draw_line_series(surface, plot_rect, data_a, plot_min, plot_max, color_a)
    draw_line_series(surface, plot_rect, data_b, plot_min, plot_max, color_b)

    val_a = data_a[-1] if data_a else 0.0
    val_b = data_b[-1] if data_b else 0.0

    # Only throttle/brake (unit="%") are stored as 0-1 fractions that
    # need scaling up for display - everything else (e.g. LiDAR meters)
    # should be shown as-is.
    display_scale = 100 if unit == "%" else 1

    label_a_surface = font_data.render(
        f"{label_a} {val_a * display_scale:.0f}{unit}", True, color_a
    )
    label_b_surface = font_data.render(
        f"{label_b} {val_b * display_scale:.0f}{unit}", True, color_b
    )

    surface.blit(label_a_surface, (rect.x + 12, rect.bottom - 20))
    surface.blit(
        label_b_surface,
        (rect.right - label_b_surface.get_width() - 12, rect.bottom - 20)
    )


def draw_scatter_chart(surface, rect, title, points,
                        x_min, x_max, x_unit,
                        y_min, y_max, y_unit,
                        center_marker=False, legend=None):
    """Generic scatter chart - used for the LiDAR top-down point cloud
    and the Radar azimuth/depth plot. `points` is a list of
    (x_value, y_value, color) tuples already in real-world units."""

    plot_rect = draw_chart_frame(
        surface, rect, title,
        y_min=y_min, y_max=y_max, y_unit=y_unit,
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
# TURN INDICATOR
# =========================================================

previous_yaw = (

    vehicle.get_transform()
    .rotation.yaw
)


indicator = "OFF"

indicator_timer = 0.0

last_time = time.time()

reverse_active = False

stuck_ticks = 0

auto_reverse_ticks_remaining = 0


# =========================================================
# MAIN LOOP
# =========================================================

try:

    while program_running:


        # =================================================
        # ADVANCE SIMULATION (synchronous mode)
        # =================================================

        world.tick()


        # =================================================
        # PYGAME CLOSE BUTTON
        # =================================================

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                program_running = False

            elif event.type == pygame.WINDOWCLOSE:

                # Closing ANY of the three windows (Data / Cameras /
                # Analytics) shuts the whole HUD down together, since
                # they all share one CARLA session underneath.
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

        # =================================================
        # SPEED
        # =================================================
        # Calculate speed from actual vehicle displacement.
        # This avoids relying on vehicle.get_velocity(), which
        # was reporting 0 in the HUD even while the car moved.

        current_location = vehicle.get_location()

        if "previous_speed_location" not in globals():
            previous_speed_location = current_location
            previous_speed_time = time.time()
            speed = 0.0
        else:
            current_time = time.time()

            distance_moved = math.sqrt(
                (current_location.x - previous_speed_location.x) ** 2
                + (current_location.y - previous_speed_location.y) ** 2
                + (current_location.z - previous_speed_location.z) ** 2
            )

            delta_speed_time = current_time - previous_speed_time

            if delta_speed_time > 0:
                speed = (distance_moved / delta_speed_time) * 3.6
            else:
                speed = 0.0

            previous_speed_location = current_location
            previous_speed_time = current_time

        # =================================================
        # HAZARD SCAN
        # =================================================

        side_distances = scan_surroundings()

        front_hazard = front_hazard_distance(side_distances)


        # =================================================
        # AUTO-REVERSE (backs up on its own if genuinely stuck,
        # not just braking momentarily)
        # =================================================

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


        # =================================================
        # REVERSE GEAR (manual R key OR automatic stuck-recovery)
        # =================================================

        with control_lock:
            reverse_requested = manual_reverse or auto_reverse_needed

        if reverse_requested and not reverse_active:

            vehicle.set_autopilot(False)

            reverse_active = True


        elif not reverse_requested and reverse_active:

            vehicle.set_autopilot(
                True,
                traffic_manager.get_port()
            )

            # Re-apply the ego's safety rules - a fresh
            # set_autopilot call resets them to TM defaults.
            traffic_manager.auto_lane_change(vehicle, True)
            traffic_manager.distance_to_leading_vehicle(vehicle, 3.0)
            traffic_manager.vehicle_percentage_speed_difference(vehicle, -10.0)
            traffic_manager.ignore_vehicles_percentage(vehicle, 0.0)
            traffic_manager.ignore_walkers_percentage(vehicle, 100.0)
            traffic_manager.ignore_lights_percentage(vehicle, 100.0)
            traffic_manager.ignore_signs_percentage(vehicle, 100.0)

            reverse_active = False


        if reverse_active:

            vehicle.apply_control(

                carla.VehicleControl(
                    throttle=0.4,
                    brake=0.0,
                    reverse=True,
                    hand_brake=False,
                    steer=0.0
                )
            )


        # =================================================
        # EMERGENCY BRAKING (forward hazard, not reversing)
        # =================================================

        emergency_braking = (
            (not reverse_active)
            and front_hazard < EMERGENCY_BRAKE_DISTANCE
        )

        if emergency_braking:

            # Override the Traffic Manager's control for this frame only -
            # forces a hard stop when something is directly ahead. We
            # don't touch autopilot itself, so it resumes normal driving
            # the instant the path is clear again.

            vehicle.apply_control(

                carla.VehicleControl(
                    throttle=0.0,
                    brake=1.0,
                    hand_brake=False,
                    steer=vehicle.get_control().steer
                )
            )


        # =================================================
        # PEDAL LEVELS (for the accelerator/brake bar HUD)
        # =================================================

        # Read from the real IMU instead of get_control(): in
        # synchronous mode get_control() can lag a tick behind the
        # Traffic Manager's actual decisions, which showed up as the
        # accelerator never registering and brake reading 100% while
        # still moving. The IMU's forward acceleration reflects what
        # the car is actually physically doing, tick for tick.

        # Read the real physical acceleration for the analytics chart.
        longitudinal_accel = sensor_readings["imu"]["accel_x"]

        # ACCEL HUD = actual throttle command, not physical acceleration.
        # A vehicle can have throttle applied while physical acceleration is
        # close to zero at steady speed.
        current_control = vehicle.get_control()
        pedal_throttle = max(0.0, min(1.0, current_control.throttle))

        # Brake remains based on actual physical deceleration.
        if longitudinal_accel < 0:
            pedal_brake = max(
                0.0,
                min(1.0, -longitudinal_accel / MAX_DECEL_FOR_BAR)
            )
        else:
            pedal_brake = 0.0


        # =================================================
        # CHART HISTORY UPDATE
        # =================================================

        speed_history.append(speed)
        accel_history.append(longitudinal_accel)
        throttle_history.append(pedal_throttle)
        brake_history.append(pedal_brake)

        # LiDAR left/right nearest-return distance, for the analytics
        # chart. LiDAR local frame: x = forward, y = right (see the
        # top-down scatter plot below). Only look at points roughly
        # alongside/ahead of the car so a return from something behind
        # doesn't get counted as a "side" hazard.
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

        location = (
            vehicle.get_location()
        )


        # =================================================
        # TURN DETECTION
        # =================================================

        current_yaw = (

            vehicle.get_transform()
            .rotation.yaw
        )


        yaw_change = (

            (
                current_yaw
                - previous_yaw
                + 180.0
            )
            % 360.0

        ) - 180.0


        current_time = time.time()


        delta_time = (

            current_time
            - last_time
        )


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

        light_state = (

            carla.VehicleLightState.Position
        )


        if indicator == "LEFT":

            light_state |= (

                carla.VehicleLightState.LeftBlinker
            )


        elif indicator == "RIGHT":

            light_state |= (

                carla.VehicleLightState.RightBlinker
            )


        vehicle.set_light_state(

            carla.VehicleLightState(
                light_state
            )
        )


        # =================================================
        # TRAFFIC LIGHT
        # =================================================

        if vehicle.is_at_traffic_light():

            state = (
                vehicle.get_traffic_light_state()
            )


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
        # TRAFFIC COUNT
        # =================================================

        alive_traffic = sum(

            1

            for npc in traffic_vehicles

            if npc is not None

            and npc.is_alive
        )


        # =================================================
        # DEMO DATA
        # =================================================

        upload = random.randint(
            50,
            150
        )


        download = random.randint(
            200,
            500
        )


        # =================================================
        # CURRENT CAMERA
        # =================================================

        with camera_lock:

            display_camera = (
                camera_mode.upper()
            )


        # =================================================
        # PYGAME - clear all three window surfaces
        # =================================================

        data_surface.fill((20, 20, 20))
        camera_surface.fill((20, 20, 20))
        chart_surface.fill((20, 20, 20))


        title = font_title.render(

            "AUTONOMOUS VEHICLE DATA",

            True,

            (255, 255, 255)
        )


        data_surface.blit(

            title,

            (35, 20)
        )


        camera_text = font_data.render(

            "Camera : "
            + display_camera,

            True,

            (100, 220, 255)
        )


        data_surface.blit(

            camera_text,

            (400, 25)
        )


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

        data_col_a = [

            "VEHICLE",

            "Tesla Model 3",

            "Driving Mode : Automatic",

            f"Speed        : {speed:.1f} km/h",

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

            f"AI           : {ai_decision}",

            "Emergency Brk: "
            + ("ACTIVE" if emergency_braking else "off"),

            "",

            "SENSOR DISTANCES (m)",

            f"Front        : {side_distances['front']:.1f}",

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

            "TRAFFIC",

            f"Vehicles     : {alive_traffic}",

            f"Light        : {traffic_light}",

            "",

            "TURN INDICATOR",

            f"Direction    : {indicator}",

            "Indicator    : "

            + (

                "ON"

                if indicator != "OFF"

                else "OFF"
            ),

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

            "ESC = Exit"
        ]


        # =================================================
        # DRAW DATA (two columns - keeps everything on screen)
        # =================================================

        sections = [

            "VEHICLE",

            "POSITION",

            "SYSTEM STATUS",

            "SENSOR DISTANCES (m)",

            "GNSS",

            "IMU",

            "COLLISION / LANE / OBSTACLE",

            "ADVANCED SENSORS",

            "TRAFFIC",

            "TURN INDICATOR",

            "DATA TRANSFER",

            "CAMERA CONTROLS"
        ]

        # The data window can get clamped shorter than DATA_WIN_HEIGHT
        # on smaller screens (see "Data window sizing" above), but the
        # text used a fixed 24px/7px line/blank spacing regardless -
        # so on a shorter screen the bottom rows (right column especially,
        # since it has the most lines) got pushed below the visible
        # window and were simply never seen. Scale the spacing down to
        # whatever actually fits in the window we ended up with, so
        # every line stays visible without the window itself growing.

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

                if text in sections:
                    text_color = (255, 210, 70)
                else:
                    text_color = (235, 235, 235)

                text_surface = font_data.render(text, True, text_color)
                data_surface.blit(text_surface, (x_pos, y_pos))

                y_pos += line_height

        draw_text_column(data_col_a, TEXT_COL_A_X)
        draw_text_column(data_col_b, TEXT_COL_B_X)


        # =================================================
        # TURN ARROW
        # =================================================

        if indicator == "LEFT":

            pygame.draw.polygon(

                data_surface,

                (255, 200, 0),

                [

                    (700, 5),

                    (650, 30),

                    (700, 55)
                ]
            )


        elif indicator == "RIGHT":

            pygame.draw.polygon(

                data_surface,

                (255, 200, 0),

                [

                    (650, 5),

                    (700, 30),

                    (650, 55)
                ]
            )


        # =================================================
        # DIRECTIONAL CAMERA THUMBNAILS + DVS  (own window)
        # =================================================
        # Clean 2x2 grid (front/right on top, left/back on bottom)
        # instead of the old "plus" layout, so the enlarged tiles
        # use the window space efficiently with no wasted gaps.

        cam_window_title = font_title.render(
            "CAMERA FEEDS", True, (255, 255, 255)
        )
        camera_surface.blit(cam_window_title, (CAMERA_MARGIN, 15))

        grid_col_x = [
            CAMERA_MARGIN,
            CAMERA_MARGIN + CAM_TILE_WIDTH + CAMERA_GRID_GAP,
        ]
        grid_row_y = [
            70,
            70 + CAM_TILE_HEIGHT + CAMERA_GRID_GAP,
        ]

        thumb_positions = {
            "front": (grid_col_x[0], grid_row_y[0]),
            "right": (grid_col_x[1], grid_row_y[0]),
            "left": (grid_col_x[0], grid_row_y[1]),
            "back": (grid_col_x[1], grid_row_y[1]),
        }

        # DVS tile sits to the right of the grid, top-aligned with it.
        dvs_x = grid_col_x[1] + CAM_TILE_WIDTH + CAMERA_GRID_GAP
        dvs_y = grid_row_y[0]

        danger_threshold = EMERGENCY_BRAKE_DISTANCE + 3.0

        for cam_name, (px, py) in thumb_positions.items():

            frame = camera_frames.get(cam_name)

            if frame is None:
                continue

            rgb = frame[:, :, :3][:, :, ::-1]

            # Captured at SENSOR_CAM_WIDTH x SENSOR_CAM_HEIGHT (the
            # sensor blueprint's fixed resolution); only rescaled to
            # CAM_TILE_WIDTH/HEIGHT if the screen was too small to
            # show tiles at full capture resolution (scale is 1.0,
            # i.e. a plain 1:1 blit, on any normal-sized display).
            cam_surface = pygame.surfarray.make_surface(
                rgb.swapaxes(0, 1)
            )
            if (CAM_TILE_WIDTH, CAM_TILE_HEIGHT) != (SENSOR_CAM_WIDTH, SENSOR_CAM_HEIGHT):
                cam_surface = pygame.transform.smoothscale(
                    cam_surface, (CAM_TILE_WIDTH, CAM_TILE_HEIGHT)
                )

            camera_surface.blit(cam_surface, (px, py))

            dist = side_distances.get(cam_name, SENSOR_SCAN_RADIUS)

            if dist < danger_threshold:
                border_color = (255, 60, 60)
            elif dist < danger_threshold * 1.6:
                border_color = (255, 210, 70)
            else:
                border_color = (90, 220, 120)

            pygame.draw.rect(
                camera_surface,
                border_color,
                (px, py, CAM_TILE_WIDTH, CAM_TILE_HEIGHT),
                3
            )

            label = font_data.render(
                f"{cam_name.upper()}  {dist:.1f} m",
                True,
                (255, 255, 255)
            )

            camera_surface.blit(label, (px + 6, py + CAM_TILE_HEIGHT - 22))

        # --- DVS event camera (own visual style, top-right tile) ---
        dvs_surface = pygame.surfarray.make_surface(
            dvs_frame_rgb.swapaxes(0, 1)
        )
        if (CAM_TILE_WIDTH, CAM_TILE_HEIGHT) != (SENSOR_CAM_WIDTH, SENSOR_CAM_HEIGHT):
            dvs_surface = pygame.transform.smoothscale(
                dvs_surface, (CAM_TILE_WIDTH, CAM_TILE_HEIGHT)
            )
        camera_surface.blit(dvs_surface, (dvs_x, dvs_y))

        pygame.draw.rect(
            camera_surface, (120, 170, 255),
            (dvs_x, dvs_y, CAM_TILE_WIDTH, CAM_TILE_HEIGHT), 3
        )

        dvs_label = font_data.render("DVS EVENT CAM", True, (255, 255, 255))
        camera_surface.blit(dvs_label, (dvs_x + 6, dvs_y + CAM_TILE_HEIGHT - 22))


        # =================================================
        # PEDAL BARS (accelerator / brake) - own block below the
        # DVS tile, with guaranteed space (was getting clipped off
        # the bottom of the old shared window).
        # =================================================

        pedal_panel_y = grid_row_y[1]

        # Fit accel/brake/speed bars into whatever horizontal room is
        # actually left in this window (which can be narrower than the
        # 1060px default on smaller screens - see CAM_TILE_WIDTH
        # scaling above), rather than assuming the default spacing
        # always has room for a third bar.
        pedal_area_width = (CAMERA_WIN_WIDTH - CAMERA_MARGIN) - (dvs_x + 20)
        base_bar_width, base_bar_gap = 70, 40
        bar_layout_scale = min(
            1.0, pedal_area_width / (3 * base_bar_width + 2 * base_bar_gap)
        )

        bar_width = max(30, int(base_bar_width * bar_layout_scale))

        bar_gap = max(15, int(base_bar_gap * bar_layout_scale))

        bar_max_height = 160

        bar_bottom = pedal_panel_y + bar_max_height

        accel_x = dvs_x + 20

        brake_x = accel_x + bar_width + bar_gap

        speed_x = brake_x + bar_width + bar_gap

        pedal_header = font_data.render("PEDALS / SPEED / GEAR", True, (255, 210, 70))
        camera_surface.blit(pedal_header, (dvs_x, pedal_panel_y - 24))

        # Outlines
        pygame.draw.rect(
            camera_surface, (90, 90, 90),
            (accel_x, pedal_panel_y, bar_width, bar_max_height), 2
        )

        pygame.draw.rect(
            camera_surface, (90, 90, 90),
            (brake_x, pedal_panel_y, bar_width, bar_max_height), 2
        )

        pygame.draw.rect(
            camera_surface, (90, 90, 90),
            (speed_x, pedal_panel_y, bar_width, bar_max_height), 2
        )

        accel_fill_height = int(bar_max_height * max(0.0, min(1.0, pedal_throttle)))

        brake_fill_height = int(bar_max_height * max(0.0, min(1.0, pedal_brake)))

        speed_fill_height = int(
            bar_max_height * max(0.0, min(1.0, speed / MAX_SPEED_FOR_BAR))
        )

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

        accel_label = font_data.render(
            f"ACCEL {pedal_throttle * 100:.0f}%", True, (200, 255, 210)
        )

        brake_label = font_data.render(
            f"BRAKE {pedal_brake * 100:.0f}%", True, (255, 200, 200)
        )

        speed_label = font_data.render(
            f"SPEED {speed:.0f}", True, (200, 235, 255)
        )

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
            gear_text,
            True,
            (255, 180, 80) if reverse_active else (200, 200, 200)
        )

        camera_surface.blit(gear_label, (accel_x, bar_bottom + 36))


        # =================================================
        # SENSOR CHARTS (two columns)
        # =================================================

        charts_title = font_title.render(
            "SENSOR ANALYTICS", True, (255, 255, 255)
        )
        chart_surface.blit(charts_title, (CHART_COL1_X, 15))

        # --- Column 1: motion over time ---
        chart_y = 50

        speed_chart_rect = pygame.Rect(
            CHART_COL1_X, chart_y, CHART_WIDTH, CHART_HEIGHT
        )

        # GNSS breadcrumb trail - lon on x, lat on y, auto-ranged to
        # whatever area has actually been driven so the path fills
        # the panel instead of being a speck on a fixed world-size axis.
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
            # fade older points, brightest at the current position
            age_frac = i / max(1, len(gnss_lon_history) - 1)
            brightness = 90 + int(age_frac * 165)
            gnss_track_points.append((lon, lat, (100, brightness, 255)))

        draw_scatter_chart(
            chart_surface, speed_chart_rect, "GNSS TRACK (lon/lat)",
            gnss_track_points,
            lon_min, lon_max, "",
            lat_min, lat_max, "",
        )

        chart_y += CHART_HEIGHT + CHART_GAP
        accel_chart_rect = pygame.Rect(
            CHART_COL1_X, chart_y, CHART_WIDTH, CHART_HEIGHT
        )
        draw_line_chart(
            chart_surface, accel_chart_rect, "IMU LONGITUDINAL ACCEL (m/s2)",
            accel_history, -8.0, 4.0, (200, 130, 255), unit=""
        )

        chart_y += CHART_HEIGHT + CHART_GAP
        pedal_chart_rect = pygame.Rect(
            CHART_COL1_X, chart_y, CHART_WIDTH, CHART_HEIGHT
        )
        draw_dual_line_chart(
            chart_surface, pedal_chart_rect, "THROTTLE vs BRAKE (%)",
            throttle_history, "Accel", (80, 220, 100),
            brake_history, "Brake", (230, 60, 60),
            0.0, 1.0, unit="%"
        )

        # --- Column 2: spatial sensors ---
        chart_y = 50

        distance_chart_rect = pygame.Rect(
            CHART_COL2_X, chart_y, CHART_WIDTH, CHART_HEIGHT
        )
        draw_dual_line_chart(
            chart_surface, distance_chart_rect, "LiDAR LEFT vs RIGHT (m)",
            lidar_left_history, "Left", (100, 200, 255),
            lidar_right_history, "Right", (255, 150, 80),
            0.0, SENSOR_SCAN_RADIUS, unit="m"
        )

        # --- LiDAR top-down point cloud ---
        chart_y += CHART_HEIGHT + CHART_GAP
        lidar_chart_rect = pygame.Rect(
            CHART_COL2_X, chart_y, CHART_WIDTH, CHART_HEIGHT
        )

        lidar_plot_points = []

        if lidar_points_raw is not None and len(lidar_points_raw) > 0:

            lidar_range = 25.0

            for point in lidar_points_raw:
                # CARLA LiDAR local frame: x = forward, y = right, z = up.
                # Plotted as a bird's-eye view with forward pointing up.
                lateral = float(point[1])
                forward = float(point[0])
                height_z = float(point[2])

                if height_z > -0.5:
                    color = (255, 210, 70)   # roughly bumper/roof height
                else:
                    color = (100, 200, 255)  # ground-level returns

                lidar_plot_points.append((lateral, forward, color))

        draw_scatter_chart(
            chart_surface, lidar_chart_rect, "LiDAR TOP-DOWN (m)",
            lidar_plot_points,
            -25.0, 25.0, "",
            -25.0, 25.0, "",
            center_marker=True
        )

        # --- Radar azimuth / depth ---
        chart_y += CHART_HEIGHT + CHART_GAP
        radar_chart_rect = pygame.Rect(
            CHART_COL2_X, chart_y, CHART_WIDTH, CHART_HEIGHT
        )

        radar_plot_points = []

        for detection in sensor_readings.get("radar", []):

            if detection["velocity"] < -0.5:
                color = (255, 80, 80)     # approaching
            elif detection["velocity"] > 0.5:
                color = (90, 220, 120)    # receding
            else:
                color = (200, 200, 200)   # roughly stationary

            radar_plot_points.append(
                (detection["azimuth_deg"], detection["depth"], color)
            )

        draw_scatter_chart(
            chart_surface, radar_chart_rect, "RADAR (deg / m)",
            radar_plot_points,
            -30.0, 30.0, "",
            0.0, 40.0, "",
            legend=[
                ("Approaching", (255, 80, 80)),
                ("Receding", (90, 220, 120)),
            ]
        )


        # =================================================
        # DISPLAY - push each window's surface to its own renderer
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

            vehicle.set_autopilot(
                False
            )


            vehicle.set_light_state(

                carla.VehicleLightState.NONE
            )


            vehicle.destroy()


    for npc in traffic_vehicles:

        if npc is not None:

            if npc.is_alive:

                npc.set_autopilot(
                    False
                )

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


    print(
        "All vehicles removed."
    )


    print(
        "Simulation stopped."
    )
