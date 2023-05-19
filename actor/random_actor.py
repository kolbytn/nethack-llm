from typing import List, Union, Tuple
import torch

from actor import LLMActor


class RandomActor(LLMActor):
    def get_action(
            self,
            lang_obs: Union[str, List[str]],
            lang_actions: List[str],
            env_actions: List[List[str]],
            return_tuple: bool = False
        ) -> Union[List[str], Tuple[List[str], str, str, int]]:

        action_weights = torch.tensor([1/8 if a.startswith("zap") or a.startswith("blow") else 1 for a in lang_actions])
        probs = torch.ones(len(env_actions)) * action_weights / torch.sum(action_weights)
        action_idx = torch.multinomial(probs, 1).item()
        env_action = env_actions[action_idx]
        lang_action = lang_actions[action_idx]
        return (env_action, lang_action, "", 0) if return_tuple else env_action
