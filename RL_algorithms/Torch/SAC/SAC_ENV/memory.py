import numpy as np

class SACBuffer():
    def __init__(self,max_size, input_shape , n_actions) -> None:
        self.mem_size = max_size
        self.ptr = 0
        self.state = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros( self.mem_size )
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward , new_state, done) -> None:
        idx = self.ptr % self.mem_size

        self.state[idx] = state
        self.new_state_memory[idx] = new_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        self.ptr+=1
    def sample_buffer(self, batch_size):
        max_mem = min(self.ptr , self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones


