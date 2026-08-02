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

    # Values to modify: episode length and player/ball speed-up
    max_steps = 1000  # episode limit (25 s at 0.025 s/step)
    max_passes = 3  # completed passes that end the episode (None: no limit)
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
    TEAMMATE_DIM = 8  # [x, y, sin(θ), cos(θ), vx, vy, vθ, role_index]
    OPPONENT_DIM = 8  # [x, y, vx, vy, vθ, role_index] (no heading observed for opponents)

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
        self.last_touch_id = None
        self.last_touch_pos = None
        self.last_touch_was_blue = False

        # Pass-to-shot tracking
        self.steps_since_pass = 0
        self.pass_pending_shot = False
        self._pass_counted_step = -1

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
        # reset per-episode opponent state (e.g. OU process)
        if self.opponent_policy is not None and hasattr(self.opponent_policy, "reset"):
            self.opponent_policy.reset()
        return super().reset(seed=seed, options=options)
    
    def step(self, action):
        """SSLBaseEnv reports the time limit as `terminated`; split it back out into
        `truncated` so PPO bootstraps V(s_T) instead of learning that the world ends
        at max_steps."""
        obs, reward, done, _, info = super().step(action)
        truncated = done and self.time_limit_reached
        # Only add episode metrics when episode ends
        # if terminated or truncated:
        info["episode_pass_count"] = self.episode_pass_count
        info["episode_goal_count"] = self.episode_goal_count
        info["episode_reward_breakdown"] = dict(self.episode_reward_breakdown)
        return obs, reward, done and not truncated, truncated, info


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
                i / max(1, self.n_robots_blue - 1),  # normalized index: 0.0 or 1.0
            )
            for i, robot in enumerate(my_robots.values())
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
                i / max(1, self.n_robots_yellow - 1),  # normalized opponent index: 0.0 or 1.0
            )
            for i, robot in enumerate(opp_robots.values())
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

        # Passing task solved: reset for a fresh spawn instead of farming the
        # same layout. Flagged like the time limit so step() reports it as a
        # truncation and PPO bootstraps V(s_T) -- otherwise cutting off the
        # future pass rewards would make the third pass look bad.
        if self.max_passes is not None and self.episode_pass_count >= self.max_passes:
            self.episode_steps = 0
            self.time_limit_reached = True
            return reward, True

        return reward, self._reward_goal() > 0  # goal ends the episode even when its reward weight is not configured

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
        else: # no blue touched the ball
            if any(robot.infrared for robot in self.frame.robots_yellow.values()): # yellow contact on ball
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
        current_dist = []  # list of distances per robot id
        last_dist = []
        closest_id = 0
        last_closest_dist = float("inf")
        for i, (current, last) in enumerate(
            zip(self.frame.robots_blue.values(), self.last_frame.robots_blue.values())
        ):
            current_dist.append(np.linalg.norm([current.x - ball.x, current.y - ball.y]))
            last_dist.append(np.linalg.norm([last.x - ball.x, last.y - ball.y]))

            # which robot was the closest in the last frame
            if last_dist[i] < last_closest_dist:
                closest_id = i
                last_closest_dist = last_dist[i]

        # reward if closest robot got closer
        delta = last_dist[closest_id] - current_dist[closest_id]
        return delta / (self.max_v * self.time_step)  # <= 1 (except for additional collisions)

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
        """Fires the step a kick happens with any forward (+x) component.

        Detects a kick as a sudden ball-speed jump (beyond drive/dribble acceleration),
        then returns 1.0 if the impulse advances the ball in +x, else 0.0.

        Independent of angle and kick strength.
        """
        if self.last_frame is None:
            return 0.0
        ball = self.frame.ball
        last = self.last_frame.ball
        dvx = ball.v_x - last.v_x
        dvy = ball.v_y - last.v_y
        dv = np.hypot(dvx, dvy)  # hypotenuse / total speed of the ball

        # ball speed above robot speed (dribbling) is a kick impulse
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
        dv = np.hypot(dvx, dvy)  # hypotenuse / total speed of the ball

        # ball speed above robot speed (dribbling) is a kick impulse
        return 1.0 if dv >= self.max_v else 0.0

    def _reward_kick_velocity(self):
        """Ball kick velocity toward the goal (x-axis), normalized by kick_speed.

        Reward kicking over dribbling (reward_progress)
        """
        # only kick speed is counted, dribbling speed returns 0. Bounded by kick_speed.
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
        """1.0 when the ball is in the goal AND was last touched in the attacking third.

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

    def _reward_out_of_bounds(self):
        """-1.0 when the ball leaves the field after a blue touch, else 0.0 (goals exempt).

        Opponent-caused outs are uncontrollable, so not penalized.
        """
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
        """Crowding penalty when >1 blue robot stacks near the ball.

        Only robots within d0 of the ball count, so off-ball spread is not rewarded.
        """
        if self.n_robots_blue < 2:
            return 0.0
        d0 = 2.0 * self.field_scale  # min comfortable gap / ball-contest radius (m)
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
            max(0.0, d0 - np.hypot(a[0] - b[0], a[1] - b[1])) # d0 - (dist between two robots)
            for a, b in itertools.combinations(near, 2) # ordered  cross-product without self-pairs
        )
        return -pen / (n_pairs * d0)

    def _reward_pass(self):
        """1.0 the step a passed ball arrives close to a teammate, else 0.0.

        Adapted from rSoccer SSLPassEnduranceEnv
        (rSoccer/rsoccer_gym/ssl/ssl_hw_challenge/pass_endurance.py,
        _calculate_reward_and_done)
        Adaptions:
        - no fixed roles/ids of robots
        - Reception = proximity, not dribbling.
        - d_min minimal ball travel distance
        """
        catch, _ = self._detect_reception()
        return catch

    def _reward_pass_forward(self):
        """Like _reward_pass but also scales by how much the pass advanced the ball toward the goal.

        A forward pass toward goal scores up to 1.0; a backward or sideways pass scores 0.0.
        Encourages passes that build toward scoring, not just any pass between teammates.
        """
        catch, direction_score = self._detect_reception()
        return catch * direction_score

    def _detect_reception(self):
        """Shared detector for _reward_pass / _reward_pass_forward.

        Returns (catch, direction_score), both 0.0 when no pass arrives this step.

        `catch` grades how well the recieving robot is able to catch the ball (low ball speed required)
        """
        if self.n_robots_blue < 2 or self.last_frame is None or self.last_touch_id is None:
            return 0.0, 0.0
        if self.last_touch_pos is None:
            return 0.0, 0.0

        d_min = 2.0 * self.field_scale  # min pass length to count (m) (should be same as minimum distance in distance reward)
        recv_radius = 0.33 * self.field_scale  # catch radius (we don't expect the robot to dribble with the ball directly)
        max_catch_speed = self.max_v  # ball faster than this cannot be brought under control

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        last_ball = np.array([self.last_frame.ball.x, self.last_frame.ball.y])

        # kick/pass must carry ball aways from passer for at least d_min
        if np.linalg.norm(ball - self.last_touch_pos) <= d_min:
            return 0.0, 0.0

        for id, robot in self.frame.robots_blue.items():
            if id == self.last_touch_id:
                continue  # passer can not pass to itself
            pos = np.array([robot.x, robot.y])
            now_near = np.linalg.norm(ball - pos) <= recv_radius
            was_near = np.linalg.norm(last_ball - pos) <= recv_radius
            if now_near and not was_near:  # ball was not near to the robot that accepts the ball
                # 1.0 for a ball arriving at rest, fading to 0.0 at max_catch_speed
                v_ball = np.hypot(self.frame.ball.v_x, self.frame.ball.v_y)
                catch = float(np.clip(1.0 - v_ball / max_catch_speed, 0.0, 1.0))
                x_gain = ball[0] - self.last_touch_pos[0]
                direction_score = float(np.clip(x_gain / self.field.length, 0.0, 1.0))
                # count the reception once per step: both rewards may be configured
                # at the same time and each calls this detector on the same frame
                if self._pass_counted_step != self.episode_steps:
                    self._pass_counted_step = self.episode_steps
                    self.episode_pass_count += 1
                self.pass_pending_shot = True
                self.steps_since_pass = 0
                return catch, direction_score
        return 0.0, 0.0

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
    
    def _reward_off_ball_positioning(self):
        """Penalize the non-closest robot for being near the ball. [MUSKAN Option 2]

        Forces exactly one robot to engage the ball while the other stays away,
        creating natural passer/receiver role differentiation.
        Returns -1.0 per step the non-closest robot is within 1.0 * field_scale of ball.
        """
        if self.n_robots_blue < 2:
            return 0.0
        ball = self.frame.ball
        dists = {rid: np.hypot(r.x - ball.x, r.y - ball.y)
                 for rid, r in self.frame.robots_blue.items()}
        closest_id = min(dists, key=dists.get)
        penalty = 0.0
        for rid, dist in dists.items():
            if rid != closest_id and dist < 1.0 * self.field_scale:
                penalty -= 1.0
        return penalty

    def _reward_receiver_positioning(self):
        """Reward the non-ball robot for being ahead of the ball in a good receiving position. [MUSKAN Option 3]

        Specifically: the non-closest robot gets reward for being:
        - Ahead of the ball in x-direction (toward goal)
        - Between 1.0 and 5.0 meters from the ball (reachable but not crowding)
        Normalized by field length so max reward = 1.0.
        """
        if self.n_robots_blue < 2:
            return 0.0
        ball = self.frame.ball
        dists = {rid: np.hypot(r.x - ball.x, r.y - ball.y)
                 for rid, r in self.frame.robots_blue.items()}
        closest_id = min(dists, key=dists.get)
        reward = 0.0
        for rid, robot in self.frame.robots_blue.items():
            if rid == closest_id:
                continue
            x_ahead = robot.x - ball.x  # positive = ahead of ball toward goal
            dist_to_ball = dists[rid]
            if x_ahead > 0 and 1.0 * self.field_scale < dist_to_ball < 5.0 * self.field_scale:
                reward += np.clip(x_ahead / self.field.length, 0.0, 1.0)
        return reward

    def _reward_kick_without_receiver(self):
        """Penalize kicking when no teammate is in a good receiving position. [MUSKAN]

        If robot A kicks but robot B is not positioned ahead of the ball,
        the kick is likely a direct shot attempt rather than a pass setup.
        Returns -1.0 when a kick happens without a receiver in position.
        """
        if self.last_frame is None or self.n_robots_blue < 2:
            return 0.0
        dvx = self.frame.ball.v_x - self.last_frame.ball.v_x
        dvy = self.frame.ball.v_y - self.last_frame.ball.v_y
        if np.hypot(dvx, dvy) < self.max_v:
            return 0.0  # no kick happened this step

        ball = self.frame.ball
        dists = {rid: np.hypot(r.x - ball.x, r.y - ball.y)
                 for rid, r in self.frame.robots_blue.items()}
        closest_id = min(dists, key=dists.get)

        for rid, robot in self.frame.robots_blue.items():
            if rid == closest_id:
                continue
            x_ahead = robot.x - ball.x
            dist = dists[rid]
            if x_ahead > 0 and 1.0 * self.field_scale < dist < 5.0 * self.field_scale:
                return 0.0  # teammate is in good position, kick is fine

        return -1.0  # kicked without a teammate in receiving position

    def _reward_kick_without_receiver_omni(self):
        """Direction-independent copy of _reward_kick_without_receiver.

        A teammate counts as receivable in any direction, not only ahead in +x.
        The lower distance bound is d_min (as in _detect_reception): a closer
        teammate cannot complete a pass because the ball would not travel far
        enough to register one.
        """
        if self.last_frame is None or self.n_robots_blue < 2:
            return 0.0
        dvx = self.frame.ball.v_x - self.last_frame.ball.v_x
        dvy = self.frame.ball.v_y - self.last_frame.ball.v_y
        if np.hypot(dvx, dvy) < self.max_v:
            return 0.0  # no kick happened this step

        ball = self.frame.ball
        dists = {rid: np.hypot(r.x - ball.x, r.y - ball.y)
                 for rid, r in self.frame.robots_blue.items()}
        closest_id = min(dists, key=dists.get)

        for rid, dist in dists.items():
            if rid == closest_id:
                continue
            if 2.0 * self.field_scale < dist < 5.0 * self.field_scale:
                return 0.0  # a teammate is at passable range, kick is fine

        return -1.0  # kicked with no teammate at passable range

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
        self._pass_counted_step = -1
        pos_frame = Frame()

        half_length = self.field.length / 2
        half_width = self.field.width / 2

        # Ball spawn position
        min_x, min_y = self.allowed_positions_ball["min"]
        max_x, max_y = self.allowed_positions_ball["max"]
        pos_frame.ball = Ball(
            x=np.random.uniform(min_x * half_length, max_x * half_length),
            y=np.random.uniform(min_y * half_width, max_y * half_width),
        )

        # Spawn one robot per configured n_robots_blue
        min_x, min_y = self.allowed_positions_blue["min"]
        max_x, max_y = self.allowed_positions_blue["max"]

        # Robot 0: attacker - spawns behind the ball
        robot0_max_x = np.clip(pos_frame.ball.x - 0.3, min_x * half_length, max_x * half_length)
        pos_frame.robots_blue[0] = Robot(
            x=np.random.uniform(min_x * half_length, robot0_max_x),
            y=np.random.uniform(min_y * half_width, max_y * half_width),
            theta=np.random.uniform(0, 360),
            )

        # Robot 1+: receiver - spawns ahead of the ball toward goal
        robot_min_x = np.clip(pos_frame.ball.x + 0.3, min_x * half_length, max_x * half_length)
        for i in range(1, self.n_robots_blue):
            pos_frame.robots_blue[i] = Robot(
                x=np.random.uniform(robot_min_x, max_x * half_length),
                y=np.random.uniform(min_y * half_width, max_y * half_width),
                theta=np.random.uniform(0, 360),
                )

        # Spawn one robot per configured n_robots_yellow (loop does nothing if 0)
        for i in range(self.n_robots_yellow):
            min_x, min_y = self.allowed_positions_yellow["min"]
            max_x, max_y = self.allowed_positions_yellow["max"]
            pos_frame.robots_yellow[i] = Robot(
                x=np.random.uniform(min_x * half_length, max_x * half_length),
                y=np.random.uniform(min_y * half_width, max_y * half_width),
                theta=np.random.uniform(0, 360),
            )

        return pos_frame
