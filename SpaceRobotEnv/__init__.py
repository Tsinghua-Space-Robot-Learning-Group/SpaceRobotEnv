import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


register(
    id="SpaceRobotState-v0",
    entry_point="SpaceRobotEnv.envs:SpaceRobotState",
    max_episode_steps=512,
)

register(
    id="SpaceRobotImage-v0",
    entry_point="SpaceRobotEnv.envs:SpaceRobotImage",
    max_episode_steps=512,
)

register(
    id="SpaceRobotDualArm-v0",
    entry_point="SpaceRobotEnv.envs:SpaceRobotDualArm",
    max_episode_steps=512,
)
