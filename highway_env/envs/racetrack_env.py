from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, Observation
from highway_env.envs.common.action import Action
from highway_env.road.lane import CircularLane, LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle


class RacetrackEnv(AbstractEnv):

    prev_dist = 0.0
    prev_lane = 'a'
    my_lane_indices = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'a'])
    total_distance = 0.0

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "OccupancyGrid",
                    "features": ["on_road"],
                    "grid_size": [[-12, 1], [-6, 7]],
                    "grid_step": [1, 1],
                    "as_image": False,
                    "align_to_vehicle_axes": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                },
                "simulation_frequency": 15,
                "policy_frequency": 5,
                "duration": 70,
                "controlled_vehicles": 1,
                "other_vehicles": 0,
                "screen_width": 600,
                "screen_height": 600,
                "centering_position": [0.5, 0.5],
                "speed_limit": 20.0,
                "track": 0
            }
        )
        return config

    def my_subsequent_lane(self, my_char):
        my_ind = np.where(self.my_lane_indices == my_char)[0][0]
        return self.my_lane_indices[my_ind + 1]

    def distance_covered(self):
        curr_dist = self.controlled_vehicles[0].lane.local_coordinates(self.controlled_vehicles[0].position)[0]
        curr_lane = self.controlled_vehicles[0].lane_index[0]

        if curr_lane == self.prev_lane:
            if (curr_dist > self.prev_dist):
                self.total_distance += (curr_dist - self.prev_dist)
        else:
            if (curr_lane == self.my_subsequent_lane(self.prev_lane)):
                self.total_distance += (curr_dist - 0.0)

        self.prev_dist = curr_dist
        self.prev_lane = curr_lane

        return self.total_distance


    def _reward(self, action: np.ndarray) -> float:
        return 0.0

    def _is_terminated(self) -> bool:
        return (not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _info(self, obs: Observation, action: Action | None = None) -> dict:
        info = super()._info(obs, action)
        info.pop("crashed")
        info.pop("action")
        info["distance_covered"] = self.distance_covered()
        info["on_road"] = self.vehicle.on_road
        return info

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        self.prev_dist = self.controlled_vehicles[0].lane.local_coordinates(self.controlled_vehicles[0].position)[0]
        self.prev_lane = self.controlled_vehicles[0].lane_index[0]
        self.total_distance = 0.0

    def _make_road(self) -> None:
        net = RoadNetwork()

        speedlimit = self.config["speed_limit"]

        if self.config["track"] == 0:

            self.my_lane_indices = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'a'])

            # Initialise First Lane
            lane = StraightLane(
                [42, 0],
                [100, 0],
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                width=5,
                speed_limit=speedlimit,
            )
            self.lane = lane

            # Straight #1
            net.add_lane("a", "b", lane)
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [42, 5],
                    [100, 5],
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )

            # 2 - Circular Arc #1
            center1 = [100, -20]
            radii1 = 20
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1,
                    np.deg2rad(90),
                    np.deg2rad(-1),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimit,
                ),
            )

            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1 + 5,
                    np.deg2rad(90),
                    np.deg2rad(-1),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )

            # 3 - Straight #2
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [120, -20],
                    [120, -30],
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )

            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [125, -20],
                    [125, -30],
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )

            # 4 - Circular Arc #2
            center2 = [105, -30]
            radii2 = 15
            net.add_lane(
                "d",
                "e",
                CircularLane(
                    center2,
                    radii2,
                    np.deg2rad(0),
                    np.deg2rad(-181),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimit,
                ),
            )

            net.add_lane(
                "d",
                "e",
                CircularLane(
                    center2,
                    radii2 + 5,
                    np.deg2rad(0),
                    np.deg2rad(-181),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )

            # 5 - Circular Arc #3
            center3 = [70, -30]
            radii3 = 15
            net.add_lane(
                "e",
                "f",
                CircularLane(
                    center3,
                    radii3 + 5,
                    np.deg2rad(0),
                    np.deg2rad(136),
                    width=5,
                    clockwise=True,
                    line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                    speed_limit=speedlimit,
                ),
            )

            net.add_lane(
                "e",
                "f",
                CircularLane(
                    center3,
                    radii3,
                    np.deg2rad(0),
                    np.deg2rad(137),
                    width=5,
                    clockwise=True,
                    line_types=(LineType.NONE, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )

            # 6 - Slant
            net.add_lane(
                "f",
                "g",
                StraightLane(
                    [55.7, -15.7],
                    [35.7, -35.7],
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )

            net.add_lane(
                "f",
                "g",
                StraightLane(
                    [59.3934, -19.2],
                    [39.3934, -39.2],
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )

            # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
            center4 = [18.1, -18.1]
            radii4 = 25
            net.add_lane(
                "g",
                "h",
                CircularLane(
                    center4,
                    radii4,
                    np.deg2rad(315),
                    np.deg2rad(170),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimit,
                ),
            )

            net.add_lane(
                "g",
                "h",
                CircularLane(
                    center4,
                    radii4 + 5,
                    np.deg2rad(315),
                    np.deg2rad(165),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )

            net.add_lane(
                "h",
                "i",
                CircularLane(
                    center4,
                    radii4,
                    np.deg2rad(170),
                    np.deg2rad(56),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimit,
                ),
            )

            net.add_lane(
                "h",
                "i",
                CircularLane(
                    center4,
                    radii4 + 5,
                    np.deg2rad(170),
                    np.deg2rad(58),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )

            # 8 - Circular Arc #5 - Reconnects to Start
            center5 = [43.2, 23.4]
            radii5 = 18.5
            net.add_lane(
                "i",
                "a",
                CircularLane(
                    center5,
                    radii5 + 5,
                    np.deg2rad(240),
                    np.deg2rad(270),
                    width=5,
                    clockwise=True,
                    line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                    speed_limit=speedlimit,
                ),
            )

            net.add_lane(
                "i",
                "a",
                CircularLane(
                    center5,
                    radii5,
                    np.deg2rad(238),
                    np.deg2rad(268),
                    width=5,
                    clockwise=True,
                    line_types=(LineType.NONE, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )



        elif self.config["track"] == 1:
            self.my_lane_indices = np.array(['a', 'b', 'c', 'd', 'a'])

            # Initialise First Lane
            lane = StraightLane(
                [42, 0],
                [100, 0],
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                width=5,
                speed_limit=speedlimit,
            )
            self.lane = lane

            # 1 Straight #1
            net.add_lane("a", "b", lane)
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [42, 5],
                    [100, 5],
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )

            # 2 - Circular Arc #1
            center1 = [100, -25]
            radii1 = 25
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1,
                    np.deg2rad(90),
                    np.deg2rad(-91),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimit,
                ),
            )
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1 + 5,
                    np.deg2rad(90),
                    np.deg2rad(-91),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )

            # 3 - Straight #2
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [100, -50],
                    [42, -50],
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [100, -55],
                    [42, -55],
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )

            # 4 - Circular Arc #2
            center2 = [42, -25]
            radii2 = 25
            net.add_lane(
                "d",
                "a",
                CircularLane(
                    center2,
                    radii2,
                    np.deg2rad(-90),
                    np.deg2rad(-271),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimit,
                ),
            )
            net.add_lane(
                "d",
                "a",
                CircularLane(
                    center2,
                    radii2 + 5,
                    np.deg2rad(-90),
                    np.deg2rad(-271),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )



        elif self.config["track"] == 2:
            self.my_lane_indices = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'a'])

            center = [0, 0]  # [m]
            radius = 25  # [m]
            alpha = 24  # [deg]
            radii = [radius, radius + 4]
            n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
            line = [[c, s], [n, c]]

            for lane in [0, 1]:
                net.add_lane(
                    "a",
                    "b",
                    CircularLane(
                        center,
                        radii[lane],
                        np.deg2rad(90 - alpha),
                        np.deg2rad(alpha),
                        clockwise=False,
                        line_types=line[lane],
                        speed_limit=speedlimit
                    ),
                )
                net.add_lane(
                    "b",
                    "c",
                    CircularLane(
                        center,
                        radii[lane],
                        np.deg2rad(alpha),
                        np.deg2rad(-alpha),
                        clockwise=False,
                        line_types=line[lane],
                        speed_limit=speedlimit
                    ),
                )
                net.add_lane(
                    "c",
                    "d",
                    CircularLane(
                        center,
                        radii[lane],
                        np.deg2rad(-alpha),
                        np.deg2rad(-90 + alpha),
                        clockwise=False,
                        line_types=line[lane],
                        speed_limit=speedlimit
                    ),
                )
                net.add_lane(
                    "d",
                    "e",
                    CircularLane(
                        center,
                        radii[lane],
                        np.deg2rad(-90 + alpha),
                        np.deg2rad(-90 - alpha),
                        clockwise=False,
                        line_types=line[lane],
                        speed_limit=speedlimit
                    ),
                )
                net.add_lane(
                    "e",
                    "f",
                    CircularLane(
                        center,
                        radii[lane],
                        np.deg2rad(-90 - alpha),
                        np.deg2rad(-180 + alpha),
                        clockwise=False,
                        line_types=line[lane],
                        speed_limit=speedlimit
                    ),
                )
                net.add_lane(
                    "f",
                    "g",
                    CircularLane(
                        center,
                        radii[lane],
                        np.deg2rad(-180 + alpha),
                        np.deg2rad(-180 - alpha),
                        clockwise=False,
                        line_types=line[lane],
                        speed_limit=speedlimit
                    ),
                )
                net.add_lane(
                    "g",
                    "h",
                    CircularLane(
                        center,
                        radii[lane],
                        np.deg2rad(180 - alpha),
                        np.deg2rad(90 + alpha),
                        clockwise=False,
                        line_types=line[lane],
                        speed_limit=speedlimit
                    ),
                )
                net.add_lane(
                    "h",
                    "a",
                    CircularLane(
                        center,
                        radii[lane],
                        np.deg2rad(90 + alpha),
                        np.deg2rad(90 - alpha),
                        clockwise=False,
                        line_types=line[lane],
                        speed_limit=speedlimit
                    ),
                )



        elif self.config["track"] == 3:
            self.my_lane_indices = np.array(['a', 'b', 'c', 'd', 'a'])

            # Initialise First Lane
            lane = StraightLane(
                [0, 0],
                [100, 0],
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                width=5,
                speed_limit=speedlimit,
            )
            self.lane = lane

            # 1 Straight #1
            net.add_lane("a", "b", lane)
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, 5],
                    [100, 5],
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )

            # 2 - Circular Arc #1
            center1 = [100, -25]
            radii1 = 25
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1,
                    np.deg2rad(90),
                    np.deg2rad(-91),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimit,
                ),
            )
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1 + 5,
                    np.deg2rad(90),
                    np.deg2rad(-91),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )

            ampli = 5
            pulse = np.pi / 50
            phase = 0

            # 3 - Sine #1
            net.add_lane(
                "c",
                "d",
                SineLane(
                    [100, -50],
                    [0, -50],
                    ampli,
                    pulse,
                    phase,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )
            net.add_lane(
                "c",
                "d",
                SineLane(
                    [100, -55],
                    [0, -55],
                    ampli,
                    pulse,
                    phase,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )

            # 4 - Circular Arc #2
            center2 = [0, -25]
            radii2 = 25
            net.add_lane(
                "d",
                "a",
                CircularLane(
                    center2,
                    radii2,
                    np.deg2rad(-90),
                    np.deg2rad(-271),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimit,
                ),
            )
            net.add_lane(
                "d",
                "a",
                CircularLane(
                    center2,
                    radii2 + 5,
                    np.deg2rad(-90),
                    np.deg2rad(-271),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )



        elif self.config["track"] == 4:
            self.my_lane_indices = np.array(['a', 'b', 'c', 'd', 'a'])

            # Initialise First Lane
            lane = StraightLane(
                [0, 0],
                [100, 0],
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                width=5,
                speed_limit=speedlimit,
            )
            self.lane = lane

            # 1 Straight #1
            net.add_lane("a", "b", lane)
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [0, 5],
                    [100, 5],
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )

            # 2 - Circular Arc #1
            center1 = [100, -25]
            radii1 = 25
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1,
                    np.deg2rad(90),
                    np.deg2rad(-91),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimit,
                ),
            )
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1 + 5,
                    np.deg2rad(90),
                    np.deg2rad(-91),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )

            ampli = 8
            pulse = np.pi / 50
            phase = 0

            # 3 - Sine #1
            net.add_lane(
                "c",
                "d",
                SineLane(
                    [100, -50],
                    [0, -50],
                    ampli,
                    pulse,
                    phase,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )
            net.add_lane(
                "c",
                "d",
                SineLane(
                    [100, -55],
                    [0, -55],
                    ampli,
                    pulse,
                    phase,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    width=5,
                    speed_limit=speedlimit,
                ),
            )

            # 4 - Circular Arc #2
            center2 = [0, -25]
            radii2 = 25
            net.add_lane(
                "d",
                "a",
                CircularLane(
                    center2,
                    radii2,
                    np.deg2rad(-90),
                    np.deg2rad(-271),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimit,
                ),
            )
            net.add_lane(
                "d",
                "a",
                CircularLane(
                    center2,
                    radii2 + 5,
                    np.deg2rad(-90),
                    np.deg2rad(-271),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimit,
                ),
            )



        elif self.config["track"] == 5:
            speedlimits = [None, 20, 20, 20, 20, 20, 20, 20, 20]
            self.my_lane_indices = np.array(['a', 'b', 'c', 'd', 'a'])

            ampli = 5
            pulse = np.pi / 20
            phase = 0

            # Initialise First Lane
            lane = SineLane(
                [0, 0],
                [100, 0],
                ampli,
                pulse,
                phase,
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                width=5,
                speed_limit=speedlimits[1],
            )
            self.lane = lane

            # Add Lanes to Road Network - Straight Section
            net.add_lane("a", "b", lane)
            net.add_lane(
                "a",
                "b",
                SineLane(
                    [0, 5],
                    [100, 5],
                    ampli,
                    pulse,
                    phase,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    width=5,
                    speed_limit=speedlimits[1],
                ),
            )

            # 2 - Circular Arc #1
            center1 = [100, -25]
            radii1 = 25
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1,
                    np.deg2rad(90),
                    np.deg2rad(-91),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimits[2],
                ),
            )
            net.add_lane(
                "b",
                "c",
                CircularLane(
                    center1,
                    radii1 + 5,
                    np.deg2rad(90),
                    np.deg2rad(-91),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimits[2],
                ),
            )

            # 3 - Sine #2
            net.add_lane(
                "c",
                "d",
                SineLane(
                    [100, -50],
                    [0, -50],
                    ampli,
                    pulse,
                    phase,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    width=5,
                    speed_limit=speedlimits[3],
                ),
            )
            net.add_lane(
                "c",
                "d",
                SineLane(
                    [100, -55],
                    [0, -55],
                    ampli,
                    pulse,
                    phase,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    width=5,
                    speed_limit=speedlimits[3],
                ),
            )

            # 4 - Circular Arc #2
            center2 = [0, -25]
            radii2 = 25
            net.add_lane(
                "d",
                "a",
                CircularLane(
                    center2,
                    radii2,
                    np.deg2rad(-90),
                    np.deg2rad(-271),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.CONTINUOUS, LineType.NONE),
                    speed_limit=speedlimits[4],
                ),
            )
            net.add_lane(
                "d",
                "a",
                CircularLane(
                    center2,
                    radii2 + 5,
                    np.deg2rad(-90),
                    np.deg2rad(-271),
                    width=5,
                    clockwise=False,
                    line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                    speed_limit=speedlimits[4],
                ),
            )



        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        rng = self.np_random

        # Controlled vehicles
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            lane_index = (
                ("a", "b", rng.integers(2))
                if i == 0
                else self.road.network.random_lane_index(rng)
            )
            controlled_vehicle = self.action_type.vehicle_class.make_on_lane(
                self.road, lane_index, speed=5.0, longitudinal=rng.uniform(20, 50)
            )

            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

        if self.config["other_vehicles"] > 0:
            # Front vehicle
            vehicle = IDMVehicle.make_on_lane(
                self.road,
                ("b", "c", lane_index[-1]),
                longitudinal=rng.uniform(
                    low=0, high=self.road.network.get_lane(("b", "c", 0)).length
                ),
                speed=6 + rng.uniform(high=3),
            )
            self.road.vehicles.append(vehicle)

            # Other vehicles
            for i in range(rng.integers(self.config["other_vehicles"])):
                random_lane_index = self.road.network.random_lane_index(rng)
                vehicle = IDMVehicle.make_on_lane(
                    self.road,
                    random_lane_index,
                    longitudinal=rng.uniform(
                        low=0, high=self.road.network.get_lane(random_lane_index).length
                    ),
                    speed=6 + rng.uniform(high=3),
                )
                # Prevent early collisions
                for v in self.road.vehicles:
                    if np.linalg.norm(vehicle.position - v.position) < 20:
                        break
                else:
                    self.road.vehicles.append(vehicle)