import io

import numpy as np
import torch
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.Render import SSLRenderField
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
import rsoccer_gym.Render.ball as render_ball
import itertools
from typing import TypeAlias, Protocol, Optional
from agent import Agent

from config import Area
from .opponent import OPPONENT_POLICIES, AgentOpponentPolicy

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
        Weighted sum of named reward functions (each normalized to [-1, 1]
        per step), configurable via the `rewards` init arg: name -> weight.
        Available names: see _reward_* methods
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

    DEFAULT_REWARD_WEIGHTS = {"proximity": 0.1, "progress": 0.8, "kick": 0.1, "passing": 0.1, "goal": 100.0}



    AreaTuple = dict[str, tuple[float, float]]
    def __init__(self, render_mode=None,
                 field_type=1,
                 n_robots_blue=2,
                 n_robots_yellow=0,
                 rewards=None,
                 allowed_positions_blue: AreaTuple = dict(),
                 allowed_positions_yellow: AreaTuple = dict(),
                 allowed_positions_ball: AreaTuple = dict(),
                 opponent_strategy: Optional[str] = None,
                 opponent_model: Optional[str] = None):
                 

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
        self.time_limit_reached = False
        self.last_touch_x = None
        self.last_touch_robot_id = None   
        self.last_touch_ball_pos = None  
        self.last_pass_detected = False 

    
        self.opponent_policy = None
        if opponent_strategy: 
            self.opponent_policy = OPPONENT_POLICIES[opponent_strategy]()

        self.field_scale = self.field.length / self.FIELD_REF_LENGTH
        # override SSLBaseEnv's motor-RPM max_v so command scaling and
        # observation normalization use the same bound
        self.max_v = self.speed_up * self.field.length / self.FIELD_CROSS_TIME
        # v_theta is in deg/s, convert MAX_W into same format
        self.max_w = np.rad2deg(self.MAX_W)
        self.kick_speed = self.KICK_SPEED_FACTOR * self.max_v
        self.max_steps = self.max_steps

        # Render based on the field_types's size
        self.field_renderer = SimFieldRenderField(self.field)
        self.window_size = self.field_renderer.window_size

        # dynamic parameters
        self.n_robots_yellow = n_robots_yellow
        self.n_robots_blue = n_robots_blue

        self.allowed_positions_blue = allowed_positions_blue
        self.allowed_positions_yellow = allowed_positions_yellow
        self.allowed_positions_ball = allowed_positions_ball
          

        # reward configuration: name -> weight, resolved to _reward_{name} methods
        self.reward_weights = dict(rewards if rewards is not None else self.DEFAULT_REWARD_WEIGHTS)
        unknown = [name for name in self.reward_weights if not callable(getattr(self, f"_reward_{name}", None))]
        if unknown:
            raise ValueError(f"unknown reward names {unknown}, available: "
                             f"{sorted(self.DEFAULT_REWARD_WEIGHTS)}")
        self.reward_functions = {name: getattr(self, f"_reward_{name}") for name in self.reward_weights}

    def set_opponent_agent(self, agent: "Agent | bytes"):
        # AsyncVectorEnv senders pass a torch.save buffer (one fd per worker) instead
        # of a live module (one fd per parameter storage); SyncVectorEnv passes the
        # module directly since nothing crosses a pipe.
        if isinstance(agent, (bytes, bytearray)):
            agent = torch.load(io.BytesIO(agent), map_location="cpu", weights_only=False)
        self.opponent_policy = AgentOpponentPolicy(agent)

    def reset(self, *, seed=None, options=None):
        # reset per-episode opponent state (e.g. OU process)
        if self.opponent_policy is not None and hasattr(self.opponent_policy, "reset"):
            self.opponent_policy.reset()
        return super().reset(seed=seed, options=options)

    def _build_obs_for(self, my_robots, opp_robots, mirror: bool):
        """mirror=True flips x and heading so the caller's team is always
        'attacking toward +x', matching how the net was trained."""
        sign = -1.0 if mirror else 1.0
        fl = self.field.length / 2
        fw = self.field.width / 2
        ball = self.frame.ball

        # TODO: understand why I don't need to turn 180 degreees
        theta_offset = 180 if mirror else 0.0

        mine = [
            (
                sign * robot.x / fl,
                sign * robot.y / fw,
                np.sin(np.deg2rad(theta_offset + robot.theta)),
                np.cos(np.deg2rad(theta_offset + robot.theta)),
                sign * robot.v_x / self.max_v,
                sign * robot.v_y / self.max_v,
                robot.v_theta / self.max_w,
            )
            for robot in my_robots.values()
        ]
        opp = [
            (
                sign * robot.x / fl,
                sign * robot.y / fw,
                np.sin(np.deg2rad(theta_offset + robot.theta)),
                np.cos(np.deg2rad(theta_offset + robot.theta)),
                sign * robot.v_x / self.max_v,
                sign * robot.v_y / self.max_v,
                robot.v_theta / self.max_w,
            )
            for robot in opp_robots.values()
        ]
        obs = np.array([
            sign * ball.x / fl,
            sign * ball.y / fw,
            sign * ball.v_x / self.kick_speed,
            sign * ball.v_y / self.kick_speed,
            *itertools.chain.from_iterable(mine),
            *itertools.chain.from_iterable(opp),
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _frame_to_observations(self):
        return self._build_obs_for(my_robots=self.frame.robots_blue,
                                   opp_robots=self.frame.robots_yellow,
                                   mirror=False)

    def _get_commands(self, action):
        # actions shape: (num_robots_blue, 5) -- one row of 5 values per robot
        commands = []

        for robot_id in range(self.n_robots_blue):
            robot_actions = action[robot_id]
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

        if self.n_robots_yellow > 0 and self.opponent_policy:
            yellow_actions = self.opponent_policy.act(self)
            for robot_id in range(self.n_robots_yellow):
                robot_actions = yellow_actions[robot_id]
                angle_rad = np.deg2rad(self.frame.robots_yellow[robot_id].theta)
                v_x = robot_actions[0] * self.max_v
                v_y = robot_actions[1] * self.max_v
                v_x_local = v_x * np.cos(angle_rad) + v_y * np.sin(angle_rad)
                v_y_local = -v_x * np.sin(angle_rad) + v_y * np.cos(angle_rad)
                commands.append(Robot(
                    yellow=True, id=robot_id,
                    v_x=v_x_local, v_y=v_y_local,
                    v_theta=robot_actions[2] * self.MAX_W,
                    kick_v_x=self.kick_speed if robot_actions[3] > 0 else 0.0,
                    dribbler=robot_actions[4] > 0,
                ))
        return commands

    def step(self, action):
        """SSLBaseEnv reports the time limit as `terminated`; split it back out into
        `truncated` so PPO bootstraps V(s_T) instead of learning that the world ends
        at max_steps."""
        observation, reward, done, _, info = super().step(action)
        truncated = done and self.time_limit_reached
        return observation, reward, done and not truncated, truncated, info

    def _calculate_reward_and_done(self):
        self.episode_steps += 1
        self._update_ball_touch()
        self.time_limit_reached = False
        if self.episode_steps >= self.max_steps:
            self.episode_steps = 0
            self.time_limit_reached = True
            return 0, True

        # End episode if robot goes out of bounds
        # if (abs(robot.x) > self.field.length / 2 or abs(robot.y) > self.field.width / 2):
        #     self.episode_steps = 0
        #     return -1, True

        reward = sum(weight * self.reward_functions[name]()
                     for name, weight in self.reward_weights.items())
        
        return reward, self._reward_goal() > 0 # goal ends the episode even when its reward weight is not configured
   
    def _update_ball_touch(self):
        for robot_id, robot in self.frame.robots_blue.items():
            if robot.infrared:
                if (self.last_touch_robot_id is not None and self.last_touch_robot_id != robot_id and self.last_touch_ball_pos is not None):
                    dist = np.linalg.norm([
                    self.frame.ball.x - self.last_touch_ball_pos[0],
                    self.frame.ball.y - self.last_touch_ball_pos[1]])
                    if dist > 0.5 * self.field_scale:
                        self.last_pass_detected = True
                self.last_touch_robot_id = robot_id
                self.last_touch_ball_pos = (self.frame.ball.x, self.frame.ball.y)

    ### REWARD DEFINITIONS
    ### should be in [-1, 1]!
    def _reward_proximity(self):
        """Step progress of the closest robot toward the ball.

        Only the closest robot to ball triggers the reward to avoid all robots trying to drive the ball
        Bounded by max_v * time_step.
        """
        if self.last_frame is None:
            return 0.0

        ball = self.frame.ball
        current_dist = []  # list of distances per robot id
        last_dist = []
        closest_id = 0
        last_closest_dist = float("inf")

        for i, (current, last) in enumerate(
                zip(self.frame.robots_blue.values(), self.last_frame.robots_blue.values())):
            current_dist.append(np.linalg.norm([current.x - ball.x, current.y - ball.y]))
            last_dist.append(np.linalg.norm([last.x - ball.x, last.y - ball.y]))

            # which robot was the closest in the last frame
            if last_dist[i] < last_closest_dist:
                closest_id = i
                last_closest_dist = last_dist[i]

        # reward if closest robot got closer
        delta = last_dist[closest_id] - current_dist[closest_id]
        return delta / (self.max_v * self.time_step) # <= 1 (except for additional collisions)

    def _reward_progress(self):
        """Ball progress toward the goal, (delta distance per step,
        normalized by the max ball displacement kick_speed * time_step)."""
        if self.last_frame is None:
            return 0.0

        #TODO: !!! ball positions go from -1 to 1 normalize them from 0 to 1
        ball = self.frame.ball
        last_ball = self.last_frame.ball
        goal_x = self.field.length / 2 #TODO: what is goal x?
        current_dist = np.linalg.norm([goal_x - ball.x, ball.y])
        last_dist = np.linalg.norm([goal_x - last_ball.x, last_ball.y])
        return (last_dist - current_dist) / (self.kick_speed * self.time_step)

    def _reward_kick(self):
        """Ball kick velocity toward the goal (x-axis),
        normalized by kick_speed (to [0, 1], negative reward is handled by reward_progress).

        Reward kicking over dribbling (reward_progress)
        """
        # only kick speed is counted, dribbling speed returns 0. Bounded by kick_speed.
        return max(0.0, (self.frame.ball.v_x - self.max_v) / (self.kick_speed - self.max_v))

    def _reward_goal(self):
        """1.0 when the ball is in the goal, else 0.0."""
        ball = self.frame.ball
        goal_x = self.field.length / 2
        in_goal = ball.x > goal_x and abs(ball.y) < self.field.goal_width / 2
        return 1.0 if in_goal else 0.0

    def _reward_goal_close(self):
        """1.0 when the ball is in the goal AND was last touched inside the last third close to the goal.

        Gives more focus to other rewards (game build-up) while the ball is farther away from the goal.
        """
        ball = self.frame.ball
        goal_x = self.field.length / 2
        in_goal = ball.x > goal_x and abs(ball.y) < self.field.goal_width / 2
        if not in_goal:
            return 0.0
        attacking_third_x = goal_x - self.field.length / 3
        touched_in_front = self.last_touch_x is not None and self.last_touch_x > attacking_third_x
        return 1.0 if touched_in_front else 0.0
    
    def _reward_passing(self):
        if self.last_pass_detected:
            self.last_pass_detected = False
            return 1.0
        return 0.0

    def _reward_time(self):
        """Constant -1.0 per step, i.e. V(s) ~= -weight * remaining steps.

        Being state- and action-independent it gives no signal *within* an
        episode; what it does is make ending the episode sooner worth more, so
        the policy prefers scoring now over stalling. Use a tiny weight: it
        accumulates over every step of the episode.
        """
        return -1.0

    def _reward_dribble(self):
        """1.0 while a teammate keeps the ball at its dribbler (infrared
        contact), else 0.0.

        Should be set to a very small factor in the total rewards.

        """
        return 1.0 if any(robot.infrared for robot in self.frame.robots_blue.values()) else 0.0

    def _get_initial_positions_frame(self):
        self.episode_steps = 0
        self.time_limit_reached = False
        self.last_touch_x = None
        self.last_touch_robot_id = None
        self.last_touch_ball_pos = None
        self.last_pass_detected = False
        pos_frame = Frame()

        goal_buffer = 1.0 * self.field_scale
        window = 2.0 * self.field_scale
        gap = 0.3

        half_length = self.field.length / 2
        half_width = self.field.width / 2
        # Ball spawn position
        min_x, min_y = self.allowed_positions_ball["min"]
        max_x, max_y = self.allowed_positions_ball["max"]
        pos_frame.ball = Ball(
            x=np.random.uniform(min_x * half_length,
                                max_x * half_length),
            y=np.random.uniform(min_y * half_width,
                                max_y * half_width),
        )

        # Spawn one robot per configured n_robots_blue
        for i in range(self.n_robots_blue):
            min_x, min_y = self.allowed_positions_blue["min"]
            max_x, max_y = self.allowed_positions_blue["max"]

            pos_frame.robots_blue[i] = Robot(
                x=np.random.uniform(min_x * half_length,
                                    max_x * half_length),
                y=np.random.uniform(min_y * half_width,
                                    max_y * half_width),
                theta=np.random.uniform(0, 360),
            )

        # Spawn one robot per configured n_robots_yellow (loop does nothing if 0)
        for i in range(self.n_robots_yellow):
            min_x, min_y = self.allowed_positions_yellow["min"]
            max_x, max_y = self.allowed_positions_yellow["max"]

            pos_frame.robots_yellow[i] = Robot(
                x=np.random.uniform(min_x * half_length,
                                    max_x * half_length),
                y=np.random.uniform(min_y * half_width,
                                    max_y * half_width),
                theta=np.random.uniform(0, 360),
            )

        return pos_frame
