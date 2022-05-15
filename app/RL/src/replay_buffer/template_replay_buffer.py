from typing import List
from ..config import config

import numpy as np

class template_replay_buffer:
    """
    replay bufferのテンプレート
    """
    def __init__(self):
        """buffer初期化"""
        self.iters_buffer = {
            "s": np.array([]),
            "a": np.array([]),
            "r": np.array([]),
            "next_s": np.array([])
            }
        self.td_errors_buffer = np.array([])
    
    def append(self, iter: dict):
        """bufferにイテレーションを追加する
        bufferサイズを超えた場合古いのから削除

        Args:
            iters (list): イテレーション dictで{s, a, r, next_s}
            td_errors (np.array): iterに対応するTD誤差
        """
        if len(self.iters_buffer["s"]) > config["buffer_size"]:
            self.iters_buffer["s"] = self.iters_buffer["s"][1:]
            self.iters_buffer["a"] = self.iters_buffer["s"][1:]
            self.iters_buffer["r"] = self.iters_buffer["r"][1:]
            self.iters_buffer["next_s"] = self.iters_buffer["next_s"][1:]
            self.td_errors_buffer = self.td_errors_buffer[1:]
            
        self.iters_buffer["s"] = np.append(self.iters_buffer, iter["s"])
        self.iters_buffer["a"] = np.append(self.iters_buffer, iter["a"])
        self.iters_buffer["r"] = np.append(self.iters_buffer, iter["r"])
        self.iters_buffer["next_s"] = np.append(self.iters_buffer, iter["next_s"])
        priority = 1 if len(self.iters_buffer["s"])==1 else max(self.td_errors_buffer) 
        self.td_errors_buffer = np.append(self.td_errors_buffer, priority)
    
    def get_iters(self, model=None) -> List[dict, np.array]:
        """
        algorithm_nameのやり方でbufferからiterationをサンプリングして返す
        Args:
            model (np.array): _description_

        Returns:
            dict: サンプリングされたiterationに対応する{s, a, r, next_s}のdict
            sampled_index: サンプリングされたiterationに対応するindex
        """
        sampled_index = np.random.choice(
            np.arange(self.iters_buffer["s"]),
            size=config["batch_size"],
            )
        return [dict(), sampled_index]
    
    def update_td_error(self, td_errors: np.array, index: np.array):
        """モデルを学習させる際に計算したtd errorを使ってbuffer内のtd errorを更新する

        Args:
            td_errors (np.array): td error
            index (np.array): buffer内に対応するindex
        """
        self.td_errors_buffer[index] = td_errors