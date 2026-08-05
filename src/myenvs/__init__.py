import pygame
from gymnasium.envs.registration import register
from rsoccer_gym.Render.robot import SSLRobot


def _colored_body(self, screen):
    """rSoccer SSLRobot.draw_robot function modified to fill robot body with the team color so it is easier to distinguish them."""
    rotated_surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)

    pygame.draw.circle(
        rotated_surface, self.team_color, (self.size, self.size), self.size
    )
    self.draw_team_tag(rotated_surface)
    self.draw_id_tag(rotated_surface)

    rotated_surface = pygame.transform.rotate(rotated_surface, -self.direction)
    new_rect = rotated_surface.get_rect(center=(self.x, self.y))

    screen.blit(rotated_surface, new_rect.topleft)


SSLRobot.draw_robot = _colored_body


register(
    id="SSLSingleRobot-v0",
    entry_point="myenvs.SingleRobot:SSLSingleRobot",
    kwargs={"render_mode": None},  # default kwargs, can be overridden in gym.make
)
register(
    id="SSLDynamicRobots-v0",
    entry_point="myenvs.DynamicRobots:SSLDynamicRobots",
    kwargs={"render_mode": None},  # default kwargs, can be overridden in gym.make
)
