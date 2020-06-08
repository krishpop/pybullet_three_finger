import gym
from gym.envs.registration import registry, make, spec


def register(id,*args,**kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id,*args,**kvargs)


register(
    id='Gripper2D-v0',
    entry_point='three_finger.envs.raw_controller_env:Gripper2DEnv',
    max_episode_steps=200,
    kwargs=dict(reward_type='dense')
)

register(
    id='Gripper2D-v1',
    entry_point='three_finger.envs.raw_controller_env:Gripper2DEnv',
    max_episode_steps=200,
    kwargs=dict(reward_type='contacts')
)

for diff_lev in ['Easy', 'Med', 'Hard']:
    for env_type in ['SamplePose', 'Goal']:
        vctr = 0
        for drop_reset in [True, False]:
            for reward_type in ['sparse', 'dense', 'contacts']:
                if env_type == 'Goal':
                    envcls = 'Gripper2DGoalEnv'
                elif diff_lev != 'Hard':
                    envcls = 'Gripper2DSamplePoseEnv'
                else:
                    envcls = 'Gripper2DHardSamplePoseEnv'
                register(
                    id='Gripper2D{}{}-v{}'.format(env_type, diff_lev, vctr),
                    entry_point='three_finger.envs.raw_controller_env:{}'.format(envcls),
                    max_episode_steps=200,
                    kwargs=dict(reward_type=reward_type, reset_on_drop=drop_reset,
                            goal_difficulty=diff_lev.lower())
                )
                vctr += 1

register(
    id='Gripper2DGoal-v0',
    entry_point='three_finger.envs.raw_controller_env:Gripper2DGoalEnv',
    max_episode_steps=200
)

register(
    id='Gripper2DGoal-v1',
    entry_point='three_finger.envs.raw_controller_env:Gripper2DGoalEnv',
    kwargs=dict(reset_on_drop=True),
    max_episode_steps=200
)

