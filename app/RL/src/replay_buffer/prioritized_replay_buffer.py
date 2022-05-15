from ..config import config
from template_replay_buffer import template_replay_buffer

import numpy as np

class prioritized_replay_buffer(template_replay_buffer):
    """
    prioritized experience replayのreplay buffer
    """
    def get_iters(self, alpha: int, beta: int) -> dict:
        """
        bufferからiterationをTD errorで重み付けしてサンプリングして返す
        Args:
            alpha (int): サンプリングする際の温度パラメータ
            beta(int): importance-sampling weightに使うパラメータ

        Returns:
            dict: サンプリングされたiterationに対応する{s, a, r, next_s}のdict
            sampled_index: サンプリングされたiterationに対応するindex
            
        """
        
        if len(self.iters_buffer["s"]) < config["batch_size"]:
            return self.iters_buffer["s"], np.arange(self.iters_buffer["s"]), np.ones(self.iters_buffer["s"])
        
        else:
            sampling_p = self.td_errors_buffer**alpha/(self.td_errors_buffer**alpha).sum()
            sampled_index = np.random.choice(
                np.arange(self.iters_buffer["s"]),
                size=config["batch_size"],
                p = sampling_p
                )
            importance_sampling_weight = (len(self.iters_buffer["s"])*sampling_p[sampled_index])**(-beta)
            importance_sampling_weight /= importance_sampling_weight.max()
            return {
            "s": self.iters_buffer["s"][sampled_index],
            "a": self.iters_buffer["a"][sampled_index],
            "r": self.iters_buffer["r"][sampled_index],
            "next_s": self.iters_buffer["next_s"][sampled_index],
            }, sampled_index, importance_sampling_weight