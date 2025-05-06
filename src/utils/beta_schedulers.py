import numpy as np 
from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal



@dataclass
class LinearScheduler:
    """
    Linear scheduler for beta values.
    """
    n_iters: int
    beta_start: float = 0.0001
    beta_end: float = 0.02
    
    def __post_init__(self):
        self.i = 0
        
    def step(self) -> None:
        """
        Returns the beta value for the current step.
        """
        self.i += 1
    
    def get(self) -> float:
        """
        Returns the beta value for the current step.
        """
        beta = self.beta_start + (self.beta_end - self.beta_start) * (self.i / (self.n_iters - 1))
        # self.i += 1
        beta = min(beta, min(self.beta_start, self.beta_end))
        beta = max(beta, max(self.beta_start, self.beta_end))   
        return beta