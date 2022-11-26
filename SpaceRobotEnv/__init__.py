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

register(
    id="SpaceRobotPointCloud-v0",
    entry_point="SpaceRobotEnv.envs:SpaceRobotPointCloud",
    max_episode_steps=512,
)

register(
    id="SpaceRobotCost-v0",
    entry_point="SpaceRobotEnv.envs:SpaceRobotCost",
    max_episode_steps=512,
)

register(
    id="SpaceRobotReorientation-v0",
    entry_point="SpaceRobotEnv.envs:SpaceRobotReorientation",
    max_episode_steps=512,
)