"""
Demonstration data collector for CIL training.

Drives CARLA's built-in autopilot across several towns and logs tuples of
(RGB image, high-level command, target waypoints) for supervised learning.

Usage:
    python planning/collect_data.py --config config/config.yaml

Output structure (in data/demonstrations/):
    episode_<N>/
        images/   <step>.png
        data.npy  numpy array of shape (T, 1 + 5*2)
                  columns: [command, wp0_x, wp0_y, ..., wp4_x, wp4_y]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _carla_vec_to_np(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=np.float32)


def collect(cfg: dict) -> None:
    import carla

    dc_cfg = cfg.get("data_collection", {})
    carla_cfg = cfg.get("carla", {})
    ego_cfg = cfg.get("ego", {})
    sensor_cfg = cfg.get("sensors", {})
    cam_cfg = sensor_cfg.get("camera", {})

    output_dir = Path(dc_cfg.get("output_dir", "data/demonstrations"))
    output_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = dc_cfg.get("num_episodes", 20)
    steps_per_episode = dc_cfg.get("steps_per_episode", 2000)
    towns = dc_cfg.get("towns", ["Town01", "Town02"])
    target_speed_kmh = dc_cfg.get("target_speed", 30.0)
    num_waypoints = cfg.get("planning", {}).get("waypoint_lookahead", 5)
    wp_spacing = cfg.get("planning", {}).get("waypoint_spacing", 2.0)

    client = carla.Client(carla_cfg.get("host", "localhost"), carla_cfg.get("port", 2000))
    client.set_timeout(carla_cfg.get("timeout", 10.0))

    for episode_idx in range(num_episodes):
        town = towns[episode_idx % len(towns)]
        print(f"\n[DataCollect] Episode {episode_idx + 1}/{num_episodes} — {town}")

        world = client.load_world(town)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = carla_cfg.get("fixed_delta_seconds", 0.05)
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter(ego_cfg.get("blueprint", "vehicle.lincoln.mkz_2017"))[0]

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        vehicle = None
        for sp in spawn_points:
            try:
                vehicle = world.spawn_actor(vehicle_bp, sp)
                break
            except Exception:
                continue

        if vehicle is None:
            print("[DataCollect] Could not spawn vehicle, skipping episode.")
            continue

        vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.set_desired_speed(vehicle, target_speed_kmh)

        # Attach RGB camera
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(cam_cfg.get("width", 800)))
        camera_bp.set_attribute("image_size_y", str(cam_cfg.get("height", 600)))
        camera_bp.set_attribute("fov", str(cam_cfg.get("fov", 90)))

        cam_pos = cam_cfg.get("position", [1.5, 0.0, 2.4])
        cam_rot = cam_cfg.get("rotation", [0.0, 0.0, 0.0])
        cam_transform = carla.Transform(
            carla.Location(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2]),
            carla.Rotation(pitch=cam_rot[0], yaw=cam_rot[1], roll=cam_rot[2]),
        )
        camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)

        latest_frame: dict = {"img": None}

        # ``latest_frame`` is bound as a default argument rather than captured by
        # closure: it is rebound once per episode, and a late-firing callback
        # from a previous episode's camera would otherwise write into the
        # *current* episode's buffer.
        def _on_image(img, latest_frame=latest_frame):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            latest_frame["img"] = arr[:, :, :3].copy()  # BGR

        camera.listen(_on_image)

        # Set up route planner for high-level commands
        carla_map = world.get_map()
        from carla import GlobalRoutePlanner
        grp = GlobalRoutePlanner(carla_map, sampling_resolution=wp_spacing)

        ep_dir = output_dir / f"episode_{episode_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        images_dir = ep_dir / "images"
        images_dir.mkdir(exist_ok=True)

        records = []  # list of (command, wp0_x, wp0_y, ..., wp4_x, wp4_y)

        # Warm up: let autopilot start moving
        for _ in range(30):
            world.tick()
            time.sleep(0.0)

        for step in range(steps_per_episode):
            world.tick()

            frame = latest_frame["img"]
            if frame is None:
                continue

            transform = vehicle.get_transform()
            location = transform.location

            # Get route waypoints ahead of the vehicle
            destination = random.choice(spawn_points).location
            try:
                route = grp.trace_route(location, destination)
            except Exception:
                continue

            if len(route) < num_waypoints + 1:
                continue

            # Extract high-level command from route[1] option
            command_enum = route[1][1]  # carla.RoadOption
            command = _road_option_to_int(command_enum)

            # Extract future waypoint positions (relative to current vehicle pose)
            waypoints_rel = _extract_relative_waypoints(
                route, transform, num_waypoints, wp_spacing
            )
            if waypoints_rel is None:
                continue

            # Save image
            img_path = images_dir / f"{step:06d}.png"
            cv2.imwrite(str(img_path), frame)

            # Build record row: [command, wx0, wy0, wx1, wy1, ..., wx4, wy4]
            row = [float(command)]
            for wp_xy in waypoints_rel:
                row.extend([float(wp_xy[0]), float(wp_xy[1])])
            records.append(row)

            if step % 200 == 0:
                print(f"  Step {step}/{steps_per_episode} — cmd={command} — {len(records)} records")

        # Save records
        if records:
            data = np.array(records, dtype=np.float32)
            np.save(str(ep_dir / "data.npy"), data)
            print(f"[DataCollect] Saved {len(records)} records to {ep_dir}")

        # Cleanup
        camera.stop()
        camera.destroy()
        vehicle.set_autopilot(False)
        vehicle.destroy()

        settings.synchronous_mode = False
        world.apply_settings(settings)

    print("\n[DataCollect] Data collection complete.")


def _road_option_to_int(option) -> int:
    """Map CARLA RoadOption enum to integer command index."""
    from carla import RoadOption
    mapping = {
        RoadOption.LANEFOLLOW: 0,
        RoadOption.LEFT: 1,
        RoadOption.RIGHT: 2,
        RoadOption.STRAIGHT: 3,
        RoadOption.CHANGELANERIGHT: 2,
        RoadOption.CHANGELANELEFT: 1,
    }
    return mapping.get(option, 0)


def _extract_relative_waypoints(
    route: list,
    vehicle_transform,
    num_waypoints: int,
    spacing: float,
) -> list | None:
    """
    Extract the next `num_waypoints` waypoints from the route and
    express their (x, y) positions in the vehicle's local frame.
    """

    vehicle_loc = vehicle_transform.location
    yaw = np.radians(vehicle_transform.rotation.yaw)
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)

    result = []
    count = 0
    for wp, _ in route[1:]:
        loc = wp.transform.location
        dx = loc.x - vehicle_loc.x
        dy = loc.y - vehicle_loc.y
        # Rotate into vehicle frame
        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy
        result.append((local_x, local_y))
        count += 1
        if count >= num_waypoints:
            break

    if len(result) < num_waypoints:
        return None
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect CIL training demonstrations from CARLA")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    collect(cfg)
