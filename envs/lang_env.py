from typing import List, Tuple
import gym
from gym import Wrapper
from nle_language_wrapper import NLELanguageWrapper

from utils.nle_utils import get_admissible, get_lang_obs, TASK_TO_DESC


class LangEnv(Wrapper):
    def __init__(self, task: str):
        self.task_id = task
        env = gym.make(
            task,
            observation_keys=("glyphs", "blstats", "tty_chars", "inv_strs", "inv_letters", "tty_cursor")
        )
        self.lang_to_action = NLELanguageWrapper(env).pre_step
        super().__init__(env)
    
    def reset(self):
        self.last_obs = super().reset()
        obs = get_lang_obs(self.last_obs, as_list=True)
        return obs
        
    def step(self, action):
        self.last_obs, reward, done, info = super().step(self.lang_to_action(action))
        obs = get_lang_obs(self.last_obs, as_list=True)
        return obs, reward, done, info
    
    def get_actions(self) -> Tuple[List[str], List[List[str]]]:
        return get_admissible(self.last_obs, allowed=self.env.actions)
        
    def get_task(self) -> str:
        return TASK_TO_DESC[self.task_id]
