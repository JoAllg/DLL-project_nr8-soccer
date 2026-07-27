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
        self.length = field.length
        self.width = field.width
        self.penalty_length = field.penalty_length
        self.penalty_width = field.penalty_width
        self.goal_width = field.goal_width
        self.goal_depth = field.goal_depth
        super().__init__()


class SSLDynamicRobots(SSLBaseEnv):
    """
    SSL Environment with dynamic number of Teammates and opponents.
    Goal: learn to kick the ball into the opponents' goal.

    Observation space: [ball_x, ball_y, ball_vx, ball_vy,
                    robot_x, robot_y, sin(θ), cos(θ), robot_vx, robot_vy, robot_vθ]
    Action space: [v_x, v_y, v_theta, kick, dribbler] (normalized to [-1, 1])

    Reward:
        Weighted sum of named reward functions (each normalized to [-1, 1]
        per step), configurable via the `rewards` init arg: name -> weight.
        Available names: see _reward_* methods
    """

    max_steps = 1000
    speed_up = 1.5

    FIELD_CROSS_TIME = 12.0
    KICK_SPEED_FACTOR = 3.8
    MAX_W = 10.0
    FIELD_REF_LENGTH = 12.0

    BALL_DIM = 4
    TEAMMATE_DIM = 7
    OPPONENT_DIM = 7

    DEFAULT_REWARD_WEIGHTS = {"proximity": 0.1, "progress": 0.8, "kick_forward": 0.1, "goal": 100.0}

    AreaTuple = dict[str, tuple[float, float]]

    def __init__(
        self,
        render_mode=None,
        field_type=1,
        n_robots_blue=2,
        n_robots_yellow=0,
        rewards=None,
        allowed_positions_blue: AreaTuple = dict(),
        allowed_positions_yellow: AreaTuple = dict(),
        allowed_positions_ball: AreaTuple = dict(),
        opponent_strategy: Optional[str] = None,
        opponent_model: Optional[str] = None,
    ):
        super().__init__(
            field_type=field_type,
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
        self.last_touch_id = None
        self.last_touch_pos = None
        self.last_touch_was_blue = False

        # Pass-to-shot tracking
        self.steps_since_pass = 0
        self.pass_pending_shot = False

        self.opponent_policy = None
        if opponent_strategy:
            self.opponent_policy = OPPONENT_POLICIES[opponent_strategy]()

        self.field_scale = self.field.length / self.FIELD_REF_LENGTH
        self.max_v = self.speed_up * self.field.length / self.FIELD_CROSS_TIME
        self.max_w = np.rad2deg(self.MAX_W)
        self.kick_speed = self.KICK_SPEED_FACTOR * self.max_v
        self.max_steps = self.max_steps

        self.field_renderer = SimFieldRenderField(self.field)
        self.window_size = self.field_renderer.window_size

        self.n_robots_yellow = n_robots_yellow
        self.n_robots_blue = n_robots_blue

        self.allowed_positions_blue = allowed_positions_blue
        self.allowed_positions_yellow = allowed_positions_yellow
        self.allowed_positions_ball = allowed_positions_ball

        self.reward_weights = dict(rewards if rewards is not None else self.DEFAULT_REWARD_WEIGHTS)
        unknown = [
            name for name in self.reward_weights if not callable(getattr(self, f"_reward_{name}", None))
        ]
        if unknown:
            raise ValueError(
                f"unknown reward names {unknown}, available: {sorted(self.DEFAULT_REWARD_WEIGHTS)}"
            )
        self.reward_functions = {name: getattr(self, f"_reward_{name}") for name in self.reward_weights}
        self.episode_reward_breakdown = {name: 0.0 for name in self.reward_weights}
        self.episode_pass_count = 0
        self.episode_goal_count = 0

    def set_opponent_agent(self, agent: "Agent | bytes"):
        # AsyncVectorEnv senders pass a torch.save buffer (one fd per worker) instead
        # of a live module (one fd per parameter storage); SyncVectorEnv passes the
        # module directly since nothing crosses a pipe.
        if isinstance(agent, (bytes, bytearray)):
            agent = torch.load(io.BytesIO(agent), map_location="cpu", weights_only=False)
        self.opponent_policy = AgentOpponentPolicy(agent)

    def reset(self, *, seed=None, options=None):
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
        obs = np.array(
            [
                sign * ball.x / fl,
                sign * ball.y / fw,
                sign * ball.v_x / self.kick_speed,
                sign * ball.v_y / self.kick_speed,
                *itertools.chain.from_iterable(mine),
                *itertools.chain.from_iterable(opp),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def _frame_to_observations(self):
        return self._build_obs_for(
            my_robots=self.frame.robots_blue, opp_robots=self.frame.robots_yellow, mirror=False
        )

    def _get_commands(self, action):
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
                    kick_v_x=self.kick_speed * max(0.0, robot_actions[3]),
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
                commands.append(
                    Robot(
                        yellow=True,
                        id=robot_id,
                        v_x=v_x_local,
                        v_y=v_y_local,
                        v_theta=robot_actions[2] * self.MAX_W,
                        kick_v_x=self.kick_speed * max(0.0, robot_actions[3]),
                        dribbler=robot_actions[4] > 0,
                    )
                )
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
        self._update_pass_shot_tracking()
        if self.episode_steps >= self.max_steps:
            self.episode_steps = 0
            self.time_limit_reached = True
            return 0, True
        
        ball = self.frame.ball
        half_length = self.field.length / 2
        half_width = self.field.width / 2

        # End episode if ball leaves the field (not a goal)
        in_goal = ball.x > half_length and abs(ball.y) < self.field.goal_width / 2
        ball_out = abs(ball.x) > half_length or abs(ball.y) > half_width

        if ball_out and not in_goal:
            self.episode_steps = 0
            return -1.0, True
        
        #reward calculation
        reward = 0.0
        for name, weight in self.reward_weights.items():
            r = self.reward_functions[name]()
            weighted = weight * r
            self.episode_reward_breakdown[name] += weighted
            reward += weighted
        return reward, self._reward_goal() > 0

    def _update_ball_touch(self):
        """Track which teammate last touched the ball and where (infrared/dribbler contact).

        last_touch_id / last_touch_pos identify the (blue) passer for _reward_pass;
        last_touch_x feeds _reward_goal_close.
        last_touch_was_blue identifies the team that last touched the ball.
        """
        for rid, robot in self.frame.robots_blue.items():
            if robot.infrared:
                self.last_touch_id = rid
                self.last_touch_x = self.frame.ball.x
                self.last_touch_pos = np.array([self.frame.ball.x, self.frame.ball.y])
                self.last_touch_was_blue = True
                break
        else:
            if any(robot.infrared for robot in self.frame.robots_yellow.values()):
                self.last_touch_was_blue = False

    def _update_pass_shot_tracking(self):
        """Track whether a shot follows a pass within a window of steps."""
        if self.pass_pending_shot:
            self.steps_since_pass += 1
            if self.last_frame is not None:
                dvx = self.frame.ball.v_x - self.last_frame.ball.v_x
                dvy = self.frame.ball.v_y - self.last_frame.ball.v_y
                if np.hypot(dvx, dvy) >= self.max_v:
                    self.pass_pending_shot = False
            if self.steps_since_pass > 15:
                self.pass_pending_shot = False

    ### REWARD DEFINITIONS
    ### should be in [-1, 1]!

    def _reward_proximity(self):
        """Step progress of the closest robot toward the ball.

        Only the closest robot to ball triggers the reward to avoid all robots trying to drive the ball.
        Bounded by max_v * time_step.
        """
        if self.last_frame is None:
            return 0.0
        ball = self.frame.ball
        current_dist = []
        last_dist = []
        closest_id = 0
        last_closest_dist = float("inf")
        for i, (current, last) in enumerate(
            zip(self.frame.robots_blue.values(), self.last_frame.robots_blue.values())
        ):
            current_dist.append(np.linalg.norm([current.x - ball.x, current.y - ball.y]))
            last_dist.append(np.linalg.norm([last.x - ball.x, last.y - ball.y]))
            if last_dist[i] < last_closest_dist:
                closest_id = i
                last_closest_dist = last_dist[i]
        delta = last_dist[closest_id] - current_dist[closest_id]
        return delta / (self.max_v * self.time_step)

    def _reward_progress(self):
        """Ball progress toward the goal, normalized by max ball displacement kick_speed * time_step."""
        if self.last_frame is None:
            return 0.0
        ball = self.frame.ball
        last_ball = self.last_frame.ball
        goal_x = self.field.length / 2
        current_dist = np.linalg.norm([goal_x - ball.x, ball.y])
        last_dist = np.linalg.norm([goal_x - last_ball.x, last_ball.y])
        return (last_dist - current_dist) / (self.kick_speed * self.time_step)

    def _reward_kick_forward(self):
        """Fires the step a kick happens with any forward (+x) component."""
        if self.last_frame is None:
            return 0.0
        ball = self.frame.ball
        last = self.last_frame.ball
        dvx = ball.v_x - last.v_x
        dvy = ball.v_y - last.v_y
        dv = np.hypot(dvx, dvy)
        if dv < self.max_v:
            return 0.0
        return 1.0 if dvx > 0 else 0.0

    def _reward_kick(self):
        """Fires for any kick in any direction."""
        if self.last_frame is None:
            return 0.0
        ball = self.frame.ball
        last = self.last_frame.ball
        dvx = ball.v_x - last.v_x
        dvy = ball.v_y - last.v_y
        dv = np.hypot(dvx, dvy)
        return 1.0 if dv >= self.max_v else 0.0

    def _reward_kick_velocity(self):
        """Ball kick velocity toward the goal (x-axis), normalized by kick_speed."""
        return max(0.0, (self.frame.ball.v_x - self.max_v) / (self.kick_speed - self.max_v))

    def _reward_goal(self):
        """1.0 when the ball is in the goal, else 0.0."""
        ball = self.frame.ball
        goal_x = self.field.length / 2
        in_goal = ball.x > goal_x and abs(ball.y) < self.field.goal_width / 2
        if in_goal:
            self.episode_goal_count += 1
            return 1.0
        return 0.0

    def _reward_goal_close(self):
        """1.0 when the ball is in the goal AND was last touched in the attacking third."""
        ball = self.frame.ball
        goal_x = self.field.length / 2
        in_goal = ball.x > goal_x and abs(ball.y) < self.field.goal_width / 2
        if not in_goal:
            return 0.0
        attacking_third_x = goal_x - self.field.length / 3
        touched_in_front = self.last_touch_x is not None and self.last_touch_x > attacking_third_x
        return 1.0 if touched_in_front else 0.0

    def _reward_out_of_bounds(self):
        """-1.0 when the ball leaves the field after a blue touch, else 0.0 (goals exempt)."""
        if not self.last_touch_was_blue:
            return 0.0
        ball = self.frame.ball
        half_length = self.field.length / 2
        half_width = self.field.width / 2
        in_goal = ball.x > half_length and abs(ball.y) < self.field.goal_width / 2
        if in_goal:
            return 0.0
        out = abs(ball.x) > half_length or abs(ball.y) > half_width
        return -1.0 if out else 0.0

    def _reward_spacing(self):
        """Crowding penalty when >1 blue robot stacks near the ball."""
        if self.n_robots_blue < 2:
            return 0.0
        d0 = 2.0 * self.field_scale
        ball = self.frame.ball
        near = [
            (r.x, r.y)
            for r in self.frame.robots_blue.values()
            if np.hypot(r.x - ball.x, r.y - ball.y) <= d0
        ]
        if len(near) < 2:
            return 0.0
        n_pairs = len(near) * (len(near) - 1) / 2
        pen = sum(
            max(0.0, d0 - np.hypot(a[0] - b[0], a[1] - b[1]))
            for a, b in itertools.combinations(near, 2)
        )
        return -pen / (n_pairs * d0)

    def _reward_pass(self):
        """1.0 the step a passed ball arrives close to a teammate, else 0.0.

        Adapted from rSoccer SSLPassEnduranceEnv.
        """
        if self.n_robots_blue < 2 or self.last_frame is None or self.last_touch_id is None:
            return 0.0
        if self.last_touch_pos is None:
            return 0.0

        d_min = 2.0 * self.field_scale
        recv_radius = 0.25 * self.field_scale

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        last_ball = np.array([self.last_frame.ball.x, self.last_frame.ball.y])

        if np.linalg.norm(ball - self.last_touch_pos) <= d_min:
            return 0.0

        for id, robot in self.frame.robots_blue.items():
            if id == self.last_touch_id:
                continue
            pos = np.array([robot.x, robot.y])
            now_near = np.linalg.norm(ball - pos) <= recv_radius
            was_near = np.linalg.norm(last_ball - pos) <= recv_radius
            if now_near and not was_near:
                self.pass_pending_shot = True
                self.steps_since_pass = 0
                return 1.0
        return 0.0

    def _reward_pass_forward(self):
        """Like _reward_pass but scales reward by how much the pass advanced the ball toward the goal.

        A forward pass toward goal scores up to 1.0; a backward or sideways pass scores 0.0.
        Encourages passes that build toward scoring, not just any pass between teammates.
        """
        if self.n_robots_blue < 2 or self.last_frame is None or self.last_touch_id is None:
            return 0.0
        if self.last_touch_pos is None:
            return 0.0

        d_min = 2.0 * self.field_scale
        recv_radius = 0.25 * self.field_scale

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        last_ball = np.array([self.last_frame.ball.x, self.last_frame.ball.y])

        if np.linalg.norm(ball - self.last_touch_pos) <= d_min:
            return 0.0

        for id, robot in self.frame.robots_blue.items():
            if id == self.last_touch_id:
                continue
            pos = np.array([robot.x, robot.y])
            now_near = np.linalg.norm(ball - pos) <= recv_radius
            was_near = np.linalg.norm(last_ball - pos) <= recv_radius
            if now_near and not was_near:
                self.episode_pass_count += 1
                x_gain = ball[0] - self.last_touch_pos[0]
                direction_score = np.clip(x_gain / self.field.length, 0.0, 1.0)
                self.pass_pending_shot = True
                self.steps_since_pass = 0
                return direction_score
        return 0.0

    def _reward_spread(self):
        """Reward robots for maintaining spatial separation from each other.

        Good positioning enables passing lanes. Normalized by field diagonal.
        Only active when n_robots_blue >= 2.
        """
        if self.n_robots_blue < 2:
            return 0.0
        positions = [np.array([r.x, r.y]) for r in self.frame.robots_blue.values()]
        max_dist = max(
            np.linalg.norm(positions[i] - positions[j])
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        )
        field_diagonal = np.hypot(self.field.length, self.field.width)
        return np.clip(max_dist / field_diagonal, 0.0, 1.0)

    def _reward_pass_to_shot(self):
        """1.0 the step a forward shot is detected within 15 steps of a completed pass.

        Rewards the full assist->shot sequence, encouraging deliberate team play
        rather than just passing for its own sake.
        """
        if not self.pass_pending_shot or self.last_frame is None:
            return 0.0
        dvx = self.frame.ball.v_x - self.last_frame.ball.v_x
        dvy = self.frame.ball.v_y - self.last_frame.ball.v_y
        if np.hypot(dvx, dvy) >= self.max_v and dvx > 0:
            self.pass_pending_shot = False
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
        """1.0 while a teammate keeps the ball at its dribbler. Use a very small weight."""
        return 1.0 if any(robot.infrared for robot in self.frame.robots_blue.values()) else 0.0

    def _get_initial_positions_frame(self):

        self.episode_reward_breakdown = {name: 0.0 for name in self.reward_weights}
        self.episode_pass_count = 0
        self.episode_goal_count = 0
        self.episode_steps = 0
        self.time_limit_reached = False
        self.last_touch_x = None
        self.last_touch_id = None
        self.last_touch_pos = None
        self.last_touch_was_blue = False
        self.steps_since_pass = 0
        self.pass_pending_shot = False
        pos_frame = Frame()

        half_length = self.field.length / 2
        half_width = self.field.width / 2

        min_x, min_y = self.allowed_positions_ball["min"]
        max_x, max_y = self.allowed_positions_ball["max"]
        pos_frame.ball = Ball(
            x=np.random.uniform(min_x * half_length, max_x * half_length),
            y=np.random.uniform(min_y * half_width, max_y * half_width),
        )

        for i in range(self.n_robots_blue):
            min_x, min_y = self.allowed_positions_blue["min"]
            max_x, max_y = self.allowed_positions_blue["max"]
            pos_frame.robots_blue[i] = Robot(
                x=np.random.uniform(min_x * half_length, max_x * half_length),
                y=np.random.uniform(min_y * half_width, max_y * half_width),
                theta=np.random.uniform(0, 360),
            )

        for i in range(self.n_robots_yellow):
            min_x, min_y = self.allowed_positions_yellow["min"]
            max_x, max_y = self.allowed_positions_yellow["max"]
            pos_frame.robots_yellow[i] = Robot(
                x=np.random.uniform(min_x * half_length, max_x * half_length),
                y=np.random.uniform(min_y * half_width, max_y * half_width),
                theta=np.random.uniform(0, 360),
            )

        return pos_frame
