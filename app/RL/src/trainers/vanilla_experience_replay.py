import numpy as np

class vanilla_experience_replay:
    
    def __init__(self, env):
        self.q_table = np.zeros((env.n_s, env.action))
        self.replay_buffer = []
        
        
    