import numpy as np 
from dataclasses import dataclass, field
from typing import Optional, Union, List, Literal



@dataclass
class LinearScheduler:
    """
    Linear scheduler for values.
    """
    n_iters: int
    start: float = 0.0001
    end: float = 0.02
    
    def __post_init__(self):
        self.i = 0
        
    def step(self) -> None:
        """
        steps the scheduler.
        """
        self.i += 1
    
    def get(self) -> float:
        """
        Returns the v value for the current step.
        """
        v = self.start + (self.end - self.start) * (self.i / (self.n_iters - 1))
        # self.i += 1
        v = max(v, min(self.start, self.end))
        v = min(v, max(self.start, self.end))   
        return v
    
@dataclass
class LogLinearScheduler:
    """
    Log linear scheduler for values.
    """
    n_iters: int
    start: float = 0.0001
    end: float = 0.02
    
    def __post_init__(self):
        self.i = 0
        
    def step(self) -> None:
        """
        steps the scheduler.
        """
        # print("stepping")
        self.i += 1
    
    def get(self) -> float:
        """
        Returns the v value for the current step.
        """
        v = np.log10(self.start) + (np.log10(self.end) - np.log10(self.start)) * (self.i / (self.n_iters - 1))
        # print(f"v: {v}")
        # self.i += 1
        v = max(v, min(np.log10(self.start), np.log10(self.end)))
        v = min(v, max(np.log10(self.start), np.log10(self.end)))   
        # print(f"v: {v}")
        return 10**v