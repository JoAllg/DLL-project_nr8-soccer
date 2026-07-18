import numpy as np
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render import SSLRenderField
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
import rsoccer_gym.Render.ball as render_ball
import itertools

render_ball.Ball.radius = 0.04  # 7x bigger for visibility


class SimFieldRenderField(SSLRenderField):
    """SSLRenderField whose dimensions come from the simulator's field
    params instead of the hardcoded 9x6 Division-B constants."""

    def __init__(self, field):
        # instance attrs shadow the class constants before
        # VSSRenderField.__init__ scales them and sizes the window
        self.length = field.length
        self.width = field.width
        self.penalty_length = field.penalty_length
        self.penalty_width = field.penalty_width
        self.goal_width = field.goal_width
        self.goal_depth = field.goal_depth
        super().__init__()


class SSLDynamicRobots(SSLBaseEnv):
    """
    SSL Environment with dynamic number of Teammates and oponnents.
    Goal learn to kick the ball into the oponnents goal.

    Observation space: [ball_x, ball_y, ball_vx, ball_vy,
                    robot_x, robot_y, sin(θ), cos(θ), robot_vx, robot_vy, robot_vθ]
    Action space: [v_x, v_y, v_theta, kick, dribbler] (normalized to [-1, 1])

    Reward:
        - Shaped: robot-to-ball proximity + ball progress toward goal (delta)
        - Goal: +100 bonus on scoring
    """

    # Values to modify: episode length and player/ball speed-up
    max_steps = 1000  # episode limit (25 s at 0.025 s/step)
    speed_up = 1.5  # multiplier on both robot and ball speed vs. realistic pace

    # Speeds anchored to real soccer: a fast pro covers the 105 m pitch in ~12 s
    # and a hard shot (~120 km/h) is ~3.8x that average run speed
    FIELD_CROSS_TIME = 12.0  # s, goal-to-goal sprint at real-player pace
    KICK_SPEED_FACTOR = 3.8  # kicked ball speed / player run speed
    MAX_W = 10.0  # rad/s

    # Field-size scaling: spawn distances below were tuned on the 12 m
    # Division-A field (field_type=0) and scale linearly with length
    FIELD_REF_LENGTH = 12.0  # m

    # Per-entity token feature widths (models.token_layout_from_env reads these);
    # must match the segments _frame_to_observations() lays out below.
    BALL_DIM = 4  # [x, y, vx, vy]
    TEAMMATE_DIM = 7  # [x, y, sin(θ), cos(θ), vx, vy, vθ]
    OPPONENT_DIM = 7  # [x, y, vx, vy, vθ] (no heading observed for opponents)

    def __init__(self, render_mode=None, field_type=1, n_robots_blue=2, n_robots_yellow=0):
        super().__init__(
            field_type=field_type,  # 0=(12x9)field, 1=(9x6)field, 2=(6x4)field
            n_robots_blue=n_robots_blue,
            n_robots_yellow=n_robots_yellow,
            time_step=0.025,
            render_mode=render_mode,
        )
        self.action_space = Box(low=-1, high=1, shape=(n_robots_blue, 5))
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=(4 + n_robots_blue * self.TEAMMATE_DIM + n_robots_yellow * self.OPPONENT_DIM,),
        )
        self.episode_steps = 0

        self.field_scale = self.field.length / self.FIELD_REF_LENGTH
        # override SSLBaseEnv's motor-RPM max_v so command scaling and
        # observation normalization use the same bound
        self.max_v = self.speed_up * self.field.length / self.FIELD_CROSS_TIME
        self.kick_speed = self.KICK_SPEED_FACTOR * self.max_v
        self.max_steps = self.max_steps

        # Render based on the field_types's size
        self.field_renderer = SimFieldRenderField(self.field)
        self.window_size = self.field_renderer.window_size

        # dynamic parameters
        self.n_robots_yellow = n_robots_yellow
        self.n_robots_blue = n_robots_blue

    def _frame_to_observations(self):
        ball = self.frame.ball
        robots_blue = self.frame.robots_blue
        robots_yellow = self.frame.robots_yellow
        fl = self.field.length / 2
        fw = self.field.width / 2

        robots_blue = [
            (
                robot.x / fl,
                robot.y / fw,
                np.sin(np.deg2rad(robot.theta)),
                np.cos(np.deg2rad(robot.theta)),
                robot.v_x / self.max_v,
                robot.v_y / self.max_v,
                robot.v_theta / self.max_w,
            )
            for robot in robots_blue.values()
        ]
        robots_yellow = [
            (
                robot.x / fl,
                robot.y / fw,
                np.sin(np.deg2rad(robot.theta)),
                np.cos(np.deg2rad(robot.theta)),
                robot.v_x / self.max_v,
                robot.v_y / self.max_v,
                robot.v_theta / self.max_w,
            )
            for robot in robots_yellow.values()
        ]

        return np.array(
            [
                ball.x / fl,
                ball.y / fw,
                ball.v_x / self.kick_speed,  # was self.max_v
                ball.v_y / self.kick_speed,  # was self.max_v
                # team robots
                *itertools.chain.from_iterable(robots_blue),
                # robots oponnents
                *itertools.chain.from_iterable(robots_yellow),
            ],
            dtype=np.float32,
        )

    def _get_commands(self, actions):
        # actions shape: (num_robots_blue, 5) -- one row of 5 values per robot
        commands = []

        for robot_id in range(self.n_robots_blue):
            robot_actions = actions[robot_id]
            angle_rad = np.deg2rad(self.frame.robots_blue[robot_id].theta)

            v_x = robot_actions[0] * self.max_v
            v_y = robot_actions[1] * self.max_v
            v_x_local = v_x * np.cos(angle_rad) + v_y * np.sin(angle_rad)
            v_y_local = -v_x * np.sin(angle_rad) + v_y * np.cos(angle_rad)

            commands.append(
                Robot(
                    yellow=False,
                    id=robot_id,
                    v_x=v_x_local,
                    v_y=v_y_local,
                    v_theta=robot_actions[2] * self.MAX_W,
                    kick_v_x=self.kick_speed if robot_actions[3] > 0 else 0.0,
                    dribbler=robot_actions[4] > 0,
                )
            )
        return commands

    def _calculate_reward_and_done(self):
        self.episode_steps += 1
        if self.episode_steps >= self.max_steps:
            self.episode_steps = 0
            return 0, True

        ball = self.frame.ball
        robots = self.frame.robots_blue
        goal_x = self.field.length / 2

        # End episode if robot goes out of bounds
        # if (abs(robot.x) > self.field.length / 2 or abs(robot.y) > self.field.width / 2):
        #     self.episode_steps = 0
        #     return -1, True

        # Reward 1: robot proximity to ball (normalized)
        # closest robot to ball proximity to avoid all robots just drive to ball
        if self.last_frame is None:
            reward_proximity = 0
        else:
            last_robots = self.last_frame.robots_blue
            current_dist = []  # list of distances per robot id
            last_dist = []
            closest_id = None
            last_closest_dist = float("inf")

            for i, (current, last) in enumerate(zip(robots.values(), last_robots.values())):
                current_dist.append(np.linalg.norm([current.x - ball.x, current.y - ball.y]))
                last_dist.append(np.linalg.norm([last.x - ball.x, last.y - ball.y]))

                # which robot was the closest in the last frame
                if last_dist[i] < last_closest_dist:
                    closest_id = i
                    last_closest_dist = last_dist[i]

            # reward if closest robot got closer
            reward_proximity = last_dist[closest_id] - current_dist[closest_id]

        # Reward 2: ball progress toward goal (delta distance, normalized)
        if self.last_frame is None:
            reward_progress = 0
        else:
            last_ball = self.last_frame.ball
            current_dist = np.linalg.norm([goal_x - ball.x, ball.y])
            last_dist = np.linalg.norm([goal_x - last_ball.x, last_ball.y])
            reward_progress = last_dist - current_dist

        reward_kick = self.frame.ball.v_x / self.kick_speed

        # Check goal condition
        if ball.x > goal_x and abs(ball.y) < self.field.goal_width / 2:
            return 100 + 0.1 * reward_proximity + 0.8 * reward_progress + 0.1 * reward_kick, True

        return 0.1 * reward_proximity + 0.8 * reward_progress + 0.1 * reward_kick, False

    def _get_initial_positions_frame(self):
        self.episode_steps = 0
        pos_frame = Frame()

        goal_buffer = 1.0 * self.field_scale
        window = 2.0 * self.field_scale
        gap = 0.3

        # Ball spawns in the attacking third, near the goal
        pos_frame.ball = Ball(
            x=np.random.uniform(self.field.length / 3, self.field.length / 2 - goal_buffer),
            y=np.random.uniform(-self.field.goal_width, self.field.goal_width),
        )

        # Spawn one robot per configured n_robots_blue
        for i in range(self.n_robots_blue):
            pos_frame.robots_blue[i] = Robot(
                x=np.random.uniform(max(0, pos_frame.ball.x - window), pos_frame.ball.x - gap),
                y=np.random.uniform(-self.field.goal_width, self.field.goal_width),
                theta=np.random.uniform(0, 360),
            )

        # Spawn one robot per configured n_robots_yellow (loop does nothing if 0)
        for i in range(self.n_robots_yellow):
            pos_frame.robots_yellow[i] = Robot(
                x=np.random.uniform(0, self.field.length / 2),
                y=np.random.uniform(-self.field.goal_width, self.field.goal_width),
                theta=np.random.uniform(0, 360),
            )

        return pos_frame
