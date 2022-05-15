from ..config import config
from template_replay_buffer import template_replay_buffer

import numpy as np

class vanilla_replay_buffer(template_replay_buffer):
    """
    vanilla experience replayのreplay buffer
    """
    def get_iters(self) -> dict:
        """
        bufferからiterationを一様分布からサンプリングして返す
        
        Returns:
            dict: サンプリングされたiterationに対応する{s, a, r, next_s}のdict
        """
        if len(self.iters_buffer["s"]) < config["batch_size"]:
            return self.iters_buffer["s"]
        
        else:
            sampled_index = np.random.choice(
                np.arange(self.iters_buffer["s"]),
                size=config["batch_size"],
                replace=False
                )
            
            return {
            "s": self.iters_buffer["s"][sampled_index],
            "a": self.iters_buffer["a"][sampled_index],
            "r": self.iters_buffer["r"][sampled_index],
            "next_s": self.iters_buffer["next_s"][sampled_index],
            }, sampled_index