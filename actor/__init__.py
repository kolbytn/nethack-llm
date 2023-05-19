from typing import List, Tuple, Union


DOMAIN_PROMPTS = {
    "nethack": "You are playing the rogue-like game NetHack.",
}


DOMAIN_AFFORDANCES = {
    "nethack": "You can move north, south, east, west, northeast, southeast, southwest, or northwest. You can attack monsters adjacent to you, pick up items under you, zap wands, eat food, wear armor, use keys, drink potions, and put on rings.",
}


class LLMActor:
    def __init__(self, domain: str = "nethack"):
        self.prompt = DOMAIN_PROMPTS[domain]
        self.affordances = DOMAIN_AFFORDANCES[domain]
        self.task = ""

    def reset(self, task_description: str = ""):
        self.task = task_description

    def get_action(
            self,
            lang_obs: Union[str, List[str]],
            lang_actions: List[str],
            env_actions: List[List[str]],
            return_tuple: bool = False
        ) -> Union[List[str], Tuple[List[str], str, str, int]]:

        raise NotImplementedError()
