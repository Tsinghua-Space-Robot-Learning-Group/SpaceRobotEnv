from os import stat
from memory import SACBuffer
from critic import CriticNetwork
from value import ValueNetwork
from actor import ActorNetwork
import torch as T
import torch.nn.functional as F

class AgentController:
    def __init__(self, model_name_critic1 , model_name_critic2, model_name_value,model_name_target_value ,model_name_actor ,
                input_shape , env ,  n_actions = 6, alpha = 0.0003 , beta = 0.0003, gamma = 0.99, 
                max_size = 1_000_000 , tau = 0.005, f1_size = 256 , f2_size = 256 , batch_size = 256,
                reward_scale = 2) -> None:
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.memory = SACBuffer(max_size= max_size,input_shape= input_shape , n_actions = n_actions )
        self.batch_size = batch_size 
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions = n_actions , input_dims=input_shape, max_actions = env.action_space.high,
                                        alpha = alpha , model_name=model_name_actor)

        self.critic1 = CriticNetwork(input_dims=input_shape ,n_actions = n_actions,  
                                        beta = beta, model_name=model_name_critic1)

        self.critic2 = CriticNetwork(input_dims = input_shape, n_actions = n_actions,  
                                        beta = beta, model_name = model_name_critic2)

        self.value = ValueNetwork(input_dims = input_shape, beta = beta, model_name = model_name_value)

        self.target_value = ValueNetwork(input_dims = input_shape, beta = beta, model_name = model_name_target_value)


        self.scale = reward_scale
        self.update_network(tau = 1) 

    def choose_action( self, obs):
        state = T.tensor([obs]).to(self.actor.device)
        action , _  = self.actor.sample_normal(state, reparamaterize=False)
        return action.cpu().detach().numpy()[0]

    def store_memory(self, state , action , reward , new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network(self, tau = None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() +\
                (1-tau) * target_value_state_dict[name].clone() 

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print("saving models")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
    def load_models(self):
        print("loading models")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()

    def learn(self):
        if self.memory.ptr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
             
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action  = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        value = self.value(state).view(-1)
        value_t = self.target_value(new_state).view(-1)
        value_t[done] = 0.0

        actions , log_probs = self.actor.sample_normal(state = state, reparamaterize = False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward( state = state, action = action)
        q2_new_policy = self.critic2.forward(state = state ,action = action )
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimiser.zero_grad() # not sure yet
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)

        value_loss.backward(retain_graph=True)
        self.value.optimiser.step()

        actions , log_probs = self.actor.sample_normal(state, reparamaterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic1.forward( state = state, action = action)
        q2_new_policy = self.critic2.forward(state = state ,action = action )
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss  = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()


        self.critic1.optimiser.zero_grad()
        self.critic2.optimiser.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_t
        q1_old_policy = self.critic1.forward(state=state, action=action).view(-1)
        q2_old_policy = self.critic2.forward(state=state, action=action).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic1.optimiser.step()
        self.critic2.optimiser.step()

        self.update_network()

