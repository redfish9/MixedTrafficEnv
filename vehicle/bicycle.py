import numpy as np
from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


class Bicycle(Vehicle):
    """
     A bicycle model with behavioral Social Force Model for non-motorized vehicles.
    """
    # basic params
    MASS: float = 1  # [kg]
    LENGTH: float = 2.0  # [m]
    WIDTH: float = 1.0  # m
    MAX_ACC: float = 5  # m/s^2
    WV: float = 1.5  # m

    # SFM params
    A_uw: float = 1.2  # boundary force
    B_uw: float = 600
    tau_d: float = 0.2  # buffer time
    A_uv: float = 0.48  # repulsive force
    B_uv: float = 600
    lam: float = 0.3  # coefficient of impact for opponent vehicles/boundaries

    # overtaking force model
    V_u: float = 5.27  # m/s
    # do not consider NMVs overtaking MVs
    ld: float = 1.5  # desired lateral spacing of bicycles surpassing bicycles
    t0: float = 1.95  # average overtaking time
    n: float = 0.7 # coefficient
    Sa: float = 2  # overtaking interval

    # following force model
    a_max: float = 1
    sigma_e: float = 2.2  # acceleration index
    bf: float = 1.61  # comfortable deceleartion
    s0: float = 9.45  # distance from bicycle to bicycle jam0 (m)
    s1: float = 6.23  # distance from bicycle to bicycle jam1 (m)
    td: float = 1.34  # safety time gap

    def __init__(
        self, road: Road, position: Vector, heading: float = 0, speed: float = 0
    ) -> None:
        super().__init__(road, position, heading, speed)
        self.lateral_speed = 0
        self.yaw_rate = 0
        self.theta = None
        self.A_lat, self.B_lat = self.lateral_lpv_dynamics()
        self.type = 0


    @property
    def state(self) -> np.ndarray:
        return np.array(
            [
                [self.position[0]],
                [self.position[1]],
                [self.heading],
                [self.speed],
                [self.lateral_speed],
                [self.yaw_rate],
            ]
        )

    def step(self, dt: float) -> None:
        self.clip_actions()
        new_state = rk4(self.derivative_func, self.state, dt=dt)  # TODO: 新状态怎么更新
        self.position = new_state[0:2, 0]
        self.heading = new_state[2, 0]
        self.speed = new_state[3, 0]
        self.lateral_speed = new_state[4, 0]
        self.yaw_rate = new_state[5, 0]

        self.on_state_update()

    def clip_actions(self) -> None:
        super().clip_actions()
        # Required because of the linearisation
        self.action["steering"] = np.clip(
            self.action["steering"], -np.pi / 2, np.pi / 2
        )
        self.yaw_rate = np.clip(
            self.yaw_rate, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED
        )
    
    def decision_model(self) -> str:
        """
        Decision mechanism for type of behaviour force.
        :return: indicator string for type of behaviour force.
        """
        leader, _ = self.road.neighbour_vehicles(self, self.lane_index)
        left_leader, _ = self.road.neighbour_vehicles(leader, leader.lane_index + 1)
        v = self.speed
        v_l = leader.speed
        d = self.lane_distance_to(leader)  # longitudinal distance between the ego vehicle and its leader
        if left_leader.position[0] - leader.position[0] > 10:  # ignore left leader 10m away
            d_left = 1000
        elif left_leader.position[0] - leader.position[0] < 2:  # check safe lane-changing spacing
            d_left = 0
        else:
            d_left = abs(leader.position[1] - left_leader.position[1])  # lateral distance
        if self.lane_index == 0:
            if (v - v_l >= 0.5) and (d < self.sa):
                if d_left >= self.WV:
                    return "overtaking"
            else:
                return "following"
        elif (v - v_l < 0.5) and (d < self.sa):
            return "following"
        else:
            return "free_movement"

    def get_behaviour_force(self) -> np.ndarray:
        """
        Calculate behaviour force beased on assigned results.
        :return: repsective behaviour force results.
        """
        b_type = self.decision_model()
        if b_type == "free_movement":
            return self.get_free_movement_force()
        elif b_type == "following":
            return self.get_following_force()
        elif b_type == "overtaking":
            return self.get_overtaking_force()

    def get_free_movement_force(self) -> np.ndarray:
        speed_difference = self.speed - self.V_u
        return np.array([self.MASS * speed_difference / self.tau_d, 0])

    def get_following_force(self) -> np.ndarray:
        """
        IDM model for car-following behaviour force.
        :return: arrary of longitudinal and lateral behaviour force.
        """
        leader = min(self.surrounding_vehicles, key=lambda v: self.distance_to(v) if self.is_in_front(v) else float('inf'))
        s = self.distance_to(leader)
        delta_v = self.speed - leader.speed
        s_star = self.s0 + max(0, self.speed * self.T + self.speed * delta_v / (2 * np.sqrt(self.a * self.b)))
        acceleration = self.a * (1 - (self.speed / self.desired_speed)**self.delta - (s_star / s)**2)
        return acceleration * np.array([np.cos(self.heading), np.sin(self.heading)])

    def get_overtaking_force(self) -> np.ndarray:
        """
        Over-taking behaviour force are calculated according to Ni.
        :return: arrary of longitudinal and lateral behaviour force.
        """
        leader, _ = self.road.neighbour_vehicles(self, self.lane_index)
        s = self.lane_distance_to(leader)
        longitudinal_force = (-0.12 * s + 0.72)
        lateral_force = max(2, longitudinal_force)
        return np.array([longitudinal_force, lateral_force])

    def get_repulsive_force(self) -> np.ndarray:
        neighbours = []
        for i in range(-1, 2):
            neighbours.append(self.road.neighbour_vehicles(self, self.lane_index + i))
        for vehicle in neighbours:
            if vehicle:
                delta = abs(self.heading - vehicle.heading)
                ld = abs(self.position[1]- vehicle.position[1])
        kuh = self.lam + (1 - self.lam) * (1 + np.cos()) / 2
        repulsive_force = self.A_uv * np.exp(- ld / self.B_uv) * self.n * kuh

    def get_social_force(self) -> np.ndarray:
        """
        Calculate the overall social force, which is the summation of 
        self drving force, repulsive force of all traffic opponents and
        boundaries, behaviour force and a disturbance force.
        """
        self_driving_force = np.array[self.MASS * self.MAX_ACC, 0]
        repulsive_force = self. get_repulsive_force()
        behaviour_force = self.get_behaviour_force()
        disturbance = np.random.rand(2)
        return self_driving_force + repulsive_force + behaviour_force + disturbance