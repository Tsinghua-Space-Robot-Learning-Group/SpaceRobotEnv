import core
import gym
import torch
from copy import deepcopy
from torch.optim import Adam
import numpy as np
import time
from memory import ReplayBuffer
import itertools
import SpaceRobotEnv
from torch.utils.tensorboard import SummaryWriter
def sac( env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:
            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================
            Calling ``pi`` should return:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs to run and train agent.
        replay_size (int): Maximum length of replay buffer.
        gamma (float): Discount factor. (Always between 0 and 1.)
        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:
            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)
        lr (float): Learning rate (used for both policy and value learning).
        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)
        batch_size (int): Minibatch size for SGD.
        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.
        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.
        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.
        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    n_update_step = 0
    time_step = 0
    n_played_games = 0 
    score_history = []
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    observation_dim = env.observation_space['observation'].shape[0]
    action_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    actor_critic_agent = actor_critic(env.observation_space['observation'], env.action_space, **ac_kwargs)
    actor_critic_agent_target = deepcopy(actor_critic_agent)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in actor_critic_agent_target.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(actor_critic_agent.q1.parameters(), actor_critic_agent.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim = observation_dim,
                                 act_dim = action_dim, 
                                 size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [actor_critic_agent.pi, 
                                                            actor_critic_agent.q1, 
                                                            actor_critic_agent.q2])
    # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        observation_, action_, reward_, new_observation_, done_ = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q_value_1 = actor_critic_agent.q1(observation_, action_)
        q_value_2 = actor_critic_agent.q2(observation_, action_)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            target_action, target_logp_a = actor_critic_agent.pi(new_observation_)

            # Target Q-values
            q1_pi_targ = actor_critic_agent_target.q1(new_observation_, target_action)
            q2_pi_targ = actor_critic_agent_target.q2(new_observation_, target_action)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = reward_ + gamma * (1 - done_) * (q_pi_targ - alpha * target_logp_a)

        # MSE loss against Bellman backup
        loss_q1 = ((q_value_1 - backup)**2).mean()
        loss_q2 = ((q_value_2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q_value_1.detach().numpy(),
                      Q2Vals=q_value_2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        observation_ = data['obs']

        pi, logp_pi = actor_critic_agent.pi(observation_)
        q1_pi = actor_critic_agent.q1(observation_, pi)
        q2_pi = actor_critic_agent.q2(observation_, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(actor_critic_agent.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
   

    def update(data, n_update_step):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **q_info)
        
        writer.add_scalar("Loss_Q", loss_q.item(), n_update_step )

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        writer.add_scalar("Loss_Pi", loss_pi.item(), n_update_step)
        # logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(actor_critic_agent.parameters(), actor_critic_agent_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(observation_, deterministic=False):
        return actor_critic_agent.act(torch.as_tensor(observation_, dtype=torch.float32), 
                      deterministic)

    def test_agent(time_step):
        print("testing model")
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            o = o['observation']
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                o = o['observation']
                ep_ret += r
                ep_len += 1
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
           
            writer.add_scalar("Test_score", ep_ret, time_step)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    observation_i, ep_ret, ep_len = env.reset(), 0, 0
    observation_i = observation_i['observation']

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            action = get_action(observation_i)
        else:
            action = env.action_space.sample()

        # Step the env
        observation_2, reward, done, _ = env.step(action)
        observation_2 = observation_2['observation']
        ep_ret += reward
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if ep_len==max_ep_len else done

        # Store experience to replay buffer
        replay_buffer.store(observation_i, action, reward, observation_2, done)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        observation_i = observation_2

        # End of trajectory handling
        if done or (ep_len == max_ep_len):
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            score_history.append(ep_ret)
            avg_score = np.mean(score_history[-100:])
            writer.add_scalar("Avg Reward", avg_score, n_played_games )
            print( 'score %.1f', ep_ret, 'avg_score %.1f' ,avg_score,'num_games', n_played_games, )
            n_played_games += 1
            observation_i, ep_ret, ep_len = env.reset(), 0, 0
            observation_i = observation_i['observation']

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                n_update_step += 1
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch , n_update_step = n_update_step)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs):
            #     logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            time_step +=1
            test_agent(time_step)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',  type=str, default='SpaceRobotState-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    # from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger_kwargs = None

    torch.set_num_threads(torch.get_num_threads())
    writer = SummaryWriter("RL_algorithms/Torch/SAC/spinup_Sac/logger")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)