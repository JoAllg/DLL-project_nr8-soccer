import numpy as np
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render import SSLRenderField
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
import rsoccer_gym.Render.ball as render_ball

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


class SSLSingleRobot(SSLBaseEnv):
    """
    Single robot SSL environment for learning to kick a ball into a goal.

    Observation space: [ball_x, ball_y, ball_vx, ball_vy,
                    robot_x, robot_y, sin(θ), cos(θ), robot_vx, robot_vy, robot_vθ]
    Action space: [v_x, v_y, v_theta, kick, dribbler] (normalized to [-1, 1])

    Reward:
        - Shaped: robot-to-ball proximity + ball progress toward goal (delta)
        - Goal: +100 bonus on scoring
    """

    # Physics constants
    MAX_V = 2.5  # m/s
    MAX_W = 10.0  # rad/s
    KICK_SPEED = 10.0  # m/s, on the reference field (scaled by field_scale)

    # Field-size scaling: the absolute distances/durations below were tuned on
    # the 12 m Division-A field (field_type=0) and scale linearly with length
    FIELD_REF_LENGTH = 12.0  # m
    MAX_STEPS_REF = 1200  # 30 s on the reference field

    # Per-entity token feature widths (models.token_layout_from_env reads these);
    # must match the segments _frame_to_observations() lays out below.
    BALL_DIM = 4  # [x, y, vx, vy]
    TEAMMATE_DIM = 7  # [x, y, sin(θ), cos(θ), vx, vy, vθ]
    OPPONENT_DIM = 5  # [x, y, vx, vy, vθ] (no heading observed for opponents)

    def __init__(self, render_mode=None, field_type=1):
        super().__init__(
            field_type=field_type,  # 0=(12x9)field, 1=(9x6)field, 2=(6x4)field
            n_robots_blue=1,
            n_robots_yellow=0,
            time_step=0.025,
            render_mode=render_mode,
        )
        self.action_space = Box(low=-1, high=1, shape=(5,))
        self.observation_space = Box(low=-1.0, high=1.0, shape=(11,))
        self.episode_steps = 0

        self.field_scale = self.field.length / self.FIELD_REF_LENGTH
        self.kick_speed = self.KICK_SPEED * self.field_scale
        self.max_steps = int(self.MAX_STEPS_REF * self.field_scale)

        # Render based on the field_types's size
        self.field_renderer = SimFieldRenderField(self.field)
        self.window_size = self.field_renderer.window_size

    def _frame_to_observations(self):
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        angle_rad = np.deg2rad(robot.theta)
        fl = self.field.length / 2
        fw = self.field.width / 2
        return np.array(
            [
                ball.x / fl,
                ball.y / fw,
                ball.v_x / self.kick_speed,  # was self.max_v
                ball.v_y / self.kick_speed,  # was self.max_v
                robot.x / fl,
                robot.y / fw,
                np.sin(angle_rad),
                np.cos(angle_rad),
                robot.v_x / self.max_v,
                robot.v_y / self.max_v,
                robot.v_theta / self.max_w,
            ],
            dtype=np.float32,
        )

    def _get_commands(self, actions):
        angle_rad = np.deg2rad(self.frame.robots_blue[0].theta)

        # Denormalize and convert from global to local robot frame
        v_x = actions[0] * self.MAX_V
        v_y = actions[1] * self.MAX_V
        v_x_local = v_x * np.cos(angle_rad) + v_y * np.sin(angle_rad)
        v_y_local = -v_x * np.sin(angle_rad) + v_y * np.cos(angle_rad)

        return [
            Robot(
                yellow=False,
                id=0,
                v_x=v_x_local,
                v_y=v_y_local,
                v_theta=actions[2] * self.MAX_W,
                kick_v_x=self.kick_speed if actions[3] > 0 else 0.0,
                dribbler=actions[4] > 0,
            )
        ]

    def _calculate_reward_and_done(self):
        self.episode_steps += 1
        if self.episode_steps >= self.max_steps:
            self.episode_steps = 0
            return 0, True

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        goal_x = self.field.length / 2

        # End episode if robot goes out of bounds
        # if (abs(robot.x) > self.field.length / 2 or abs(robot.y) > self.field.width / 2):
        #     self.episode_steps = 0
        #     return -1, True

        # Reward 1: robot proximity to ball (normalized)
        if self.last_frame is None:
            reward_proximity = 0
        else:
            last_robot = self.last_frame.robots_blue[0]
            last_ball = self.last_frame.ball
            current_dist = np.linalg.norm([robot.x - ball.x, robot.y - ball.y])
            last_dist = np.linalg.norm(
                [last_robot.x - last_ball.x, last_robot.y - last_ball.y]
            )
            reward_proximity = last_dist - current_dist

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
            return (
                100
                + 0.1 * reward_proximity
                + 0.8 * reward_progress
                + 0.1 * reward_kick,
                True,
            )

        return 0.1 * reward_proximity + 0.8 * reward_progress + 0.1 * reward_kick, False

    def _get_initial_positions_frame(self):
        self.episode_steps = 0
        pos_frame = Frame()

        # Task-layout distances scale with the field; the robot-ball gap is a
        # physical clearance (robot radius 0.09 m) and stays absolute
        goal_buffer = 1.0 * self.field_scale  # spawn band distance from goal line
        window = 2.0 * self.field_scale  # how far behind the ball the robot starts
        gap = 0.3

        # Ball spawns in the attacking third, near the goal
        pos_frame.ball = Ball(
            x=np.random.uniform(
                self.field.length / 3, self.field.length / 2 - goal_buffer
            ),
            y=np.random.uniform(-self.field.goal_width, self.field.goal_width),
        )

        # Robot spawns behind the ball (lower x) in the same area
        pos_frame.robots_blue[0] = Robot(
            x=np.random.uniform(
                max(0, pos_frame.ball.x - window), pos_frame.ball.x - gap
            ),
            y=np.random.uniform(-self.field.goal_width, self.field.goal_width),
            theta=np.random.uniform(0, 360),
        )

        return pos_frame
