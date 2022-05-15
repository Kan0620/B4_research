from ..config import config
from template_replay_buffer import template_replay_buffer

import numpy as np
from torch.nn import Model

class large_batch_replay_buffer(template_replay_buffer):
    """
    large batch experience replayのreplay buffer
    """
    def get_iters(self, model: Model) -> dict:
        """
        bufferからiterationを近似したTD errorで重み付けしてサンプリングして返す
        
        Returns:
            dict: サンプリングされたiterationに対応する{s, a, r, next_s}のdict
        """
        if len(self.iters_buffer["s"]) < config["batch_size"]:
            return self.iters_buffer["s"], np.arange(self.iters_buffer["s"]), np.ones(self.iters_buffer["s"])
        
        model.eval()
        
        return dict()