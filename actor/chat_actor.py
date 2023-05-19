from typing import List, Union, Tuple, Dict, Any
import torch
import json

from actor import LLMActor
from utils.gpt_utils import get_chat
from utils.nle_utils import TASK_TO_DESC


class ChatActor(LLMActor):
    def __init__(self, fewshot=4, use_cot=True, **kwargs):
        super().__init__(**kwargs)
        self.fewshot = fewshot
        self.use_cot = use_cot

    def reset(self, task_description: str = ""):
        return super().reset(task_description)
    
    def _get_fewshot_actor_prompt(
            self,
            task: str,
            state: List[str],
            admissible: List[str]
        ) -> List[str]:

        turns = []
        examples = NLE_EXMAPLES[:self.fewshot]
        for example in examples:

            turns.append("{}Game Description:\n{}\n\nChoose the best action.\n{}".format(
                "Your task is to {}\n\n".format(
                    TASK_TO_DESC[example["task_id"]]
                ),
                "\n".join(example["state"]),
                "\n".join(["{}) {}".format(chr(ord('A') + i), a) for i, a in enumerate(example["admissible"])]),
            ))

            turns.append("{}I choose to: {}) {}".format(
                "{}\n\n".format(example["act_explanation"]) if self.use_cot else "",
                chr(ord('A') + example["admissible"].index(example["action"])), 
                example["action"]
            ))

        turns.append("Your task is to {}\n\nGame Description:\n{}\n\nChoose the best action.\n{}".format(
            task,
            "\n".join(state),
            "\n".join(["{}) {}".format(chr(ord('A') + i), a) for i, a in enumerate(admissible)]),
        ))

        return turns

    def _get_score(
            self,
            summary: Union[str, List[str]],
            actions: Union[str, List[str]], 
            task: str = None
        ) -> Tuple[torch.Tensor, str, int]:
        if task is None:
            task = self.task
        if isinstance(actions, str):
            actions = [actions]
        if isinstance(summary, str):
            summary = [x for x in summary.split(". ")]

        turns = self._get_fewshot_actor_prompt(
            task,
            summary,
            actions
        )

        out, tokens = get_chat(turns, system_message=self.prompt + " " + self.affordances)
        
        predicted = out
        action_start_idx = predicted.find("I choose to:")
        action_end_idx = len(predicted)
        if action_start_idx > -1:
            action_start_idx += len("I choose to:")+4
            action_end_idx = [
                predicted[action_start_idx:].find("\n"),
                predicted[action_start_idx:].find(","),
                predicted[action_start_idx:].find("."),
                predicted[action_start_idx:].find(" and")
            ]
            action_end_idx = min([len(predicted)] + [action_start_idx + x for x in action_end_idx if x > -1])
            predicted = predicted[action_start_idx:action_end_idx]

        scores = torch.tensor([
            sum(int(token in a.strip().lower().split())
            for token in predicted.strip().lower().split()) / len(a.strip().split())
            for a in actions
        ], dtype=torch.float32)
        scores[scores == 0] = -torch.inf

        return scores, out, tokens

    def get_action(
            self,
            lang_obs: Union[str, List[str]],
            lang_actions: List[str],
            env_actions: List[List[str]],
            return_tuple: bool = False
        ) -> Union[List[str], Tuple[List[str], str, str, int]]:

        # Get scores for high actions
        scores, generation, tokens = self._get_score(lang_obs, lang_actions)

        if torch.all(scores == -torch.inf):
            scores = torch.ones_like(scores)
        else:
            max_score = torch.max(scores)
            scores = torch.where(scores == max_score, 1, -torch.inf)
        probs = torch.softmax(scores, 0)
        action_idx = torch.multinomial(probs, 1).item()
        lang_action = lang_actions[action_idx]

        if return_tuple:
            return env_actions[lang_actions.index(lang_action)], lang_action, generation, tokens
        else:
            return env_actions[lang_actions.index(lang_action)]


NLE_EXMAPLES = [
    {
        "task_id": "MiniHack-LavaCross-Levitate-Ring-Inv-v0",
        "state": [
            "You have a +1 club (weapon in hand)",
            "You have a blessed +2 sling (alternate weapon; not wielded)",
            "You have 17 uncursed flint stones (in quiver pouch)",
            "You have 30 uncursed rocks",
            "You have an uncursed +0 leather armor (being worn)",
            "You have a ring of levitation (on right hand)",
            "Strength: 20/19",
            "Dexterity: 11",
            "Constitution: 18",
            "Intelligence: 9",
            "Wisdom: 7",
            "Charisma: 10",
            "Depth: 1",
            "Gold: 0",
            "HP: 16/16",
            "Energy: 2/2",
            "AC: 8",
            "XP: 1/0",
            "Time: 3",
            "Position: 35|10",
            "Hunger: Not Hungry",
            "Monster Level: 0",
            "Encumbrance: Unencumbered",
            "Dungeon Number: 0",
            "Level Number: 1",
            "Score: 0",
            "Alignment: Lawful",
            "Condition: Levitating",
            "You see a vertical wall far east",
            "You see a horizontal wall near north, northeast, southeast, and south",
            "You see a lava near east",
            "You see a area of lava near east northeast and east southeast",
            "You see a stairs down near east southeast",
            "You see a southwest corner near southwest",
            "You see a vertical wall near west",
            "You see a northwest room corner near northwest",
            "You see a stairs up adjacent northwest."
        ],
        "admissible": ["move north", "move south", "move east", "move west", "move northwest", "move northeast", "move southwest", "move southeast"],
        "action": "move southeast",
        "act_explanation": "The first step of my task is to put on the ring. I already have a ring on my right hand. The second step of my task is to cross the lava to the stairs down. There is lava northeast, east, and southeast. The stairs down are southeast. I should move southeast."
    },
    {
        "task_id": "MiniHack-Eat-v0",
        "state": [
            "You have a +1 club (weapon in hand)",
            "You have a +2 sling (alternate weapon; not wielded)",
            "You have 12 uncursed flint stones (in quiver pouch)",
            "You have 30 uncursed rocks",
            "You have an uncursed +0 leather armor (being worn)",
            "Strength: 21/19",
            "Dexterity: 17",
            "Constitution: 14",
            "Intelligence: 8",
            "Wisdom: 7",
            "Charisma: 8",
            "Depth: 1",
            "Gold: 0",
            "HP: 16/16",
            "Energy: 2/2",
            "AC: 8",
            "XP: 1/0",
            "Time: 2",
            "Position: 38|11",
            "Hunger: Not Hungry",
            "Monster Level: 0",
            "Encumbrance: Unencumbered",
            "Dungeon Number: 0",
            "Level Number: 1",
            "Score: 0",
            "Alignment: Neutral",
            "Condition: None",
            "You see a dark area near north, northeast, east, southeast, south, southwest, west, and northwest",
            "You see a stairs up adjacent east",
            "You see a apple adjacent west."
        ],
        "admissible": ["move north", "move south", "move east", "move west", "move northwest", "move northeast", "move southwest", "move southeast"],
        "action": "move west",
        "act_explanation": "The first step of my task is to pick up the apple. There is an apple to the west. I can only pick up items that are under me. I should move west."
    },
    {
        "task_id": "MiniHack-Wear-v0",
        "state": [
            "You have a +1 club (weapon in hand)",
            "You have a +2 sling (alternate weapon; not wielded)",
            "You have 23 uncursed flint stones (in quiver pouch)",
            "You have 24 uncursed rocks",
            "You have an uncursed +0 leather armor (being worn)",
            "Strength: 22/19",
            "Dexterity: 10",
            "Constitution: 17",
            "Intelligence: 10",
            "Wisdom: 8",
            "Charisma: 8",
            "Depth: 1",
            "Gold: 0",
            "HP: 16/16",
            "Energy: 2/2",
            "AC: 8",
            "XP: 1/0",
            "Time: 5",
            "Position: 40|11",
            "Hunger: Not Hungry",
            "Monster Level: 0",
            "Encumbrance: Unencumbered",
            "Dungeon Number: 0",
            "Level Number: 1",
            "Score: 0",
            "Alignment: Lawful",
            "Condition: None",
            "You see a dark area near north, south, southwest, west, and northwest",
            "You see a stairs up near west southwest",
            "You see a dark area adjacent northeast, east, and southeast",
            "You see here a robe."
        ],
        "admissible": ["move north", "move south", "move east", "move west", "move northwest", "move northeast", "move southwest", "move southeast", "pick up a robe"],
        "action": "pick up a robe",
        "act_explanation": "The first step of my task is to pick up the robe. The description doesn't mention having a robe yet. There is a robe here. I can pick up items that are under me. I should pick up the robe."
    },
    {
        "task_id": "MiniHack-LavaCross-Levitate-Potion-Inv-v0",
        "state": [
            "You have a +1 club (weapon in hand)",
            "You have a +2 sling (alternate weapon; not wielded)",
            "You have 19 uncursed flint stones (in quiver pouch)",
            "You have 29 uncursed rocks",
            "You have an uncursed +0 leather armor (being worn)",
            "Strength: 18/18",
            "Dexterity: 15",
            "Constitution: 16",
            "Intelligence: 7",
            "Wisdom: 9",
            "Charisma: 10",
            "Depth: 1",
            "Gold: 0",
            "HP: 16/16",
            "Energy: 2/2",
            "AC: 8",
            "XP: 1/0",
            "Time: 7",
            "Position: 39|8",
            "Hunger: Not Hungry",
            "Monster Level: 0",
            "Encumbrance: Unencumbered",
            "Dungeon Number: 0",
            "Level Number: 1",
            "Score: 10",
            "Alignment: Neutral",
            "Condition: Levitating",
            "You see a vertical wall far west",
            "You see a vertical wall near east",
            "You see a southeast corner near southeast",
            "You see a horizontal wall near south and southwest",
            "You see a area of lava near south southwest",
            "You see a stairs up near west southwest",
            "You see a stairs down very near east",
            "You see a lava very near south southwest",
            "You see a horizontal wall adjacent north, northeast, and northwest",
            "You see a lava adjacent southwest and west"
        ],
        "admissible": ["move north", "move south", "move east", "move west", "move northwest", "move northeast", "move southwest", "move southeast"],
        "action": "move east",
        "act_explanation": "The first step of my task is to drink the potion. The description doesn't mention a potion, so I already drank it. The next step is cross the lava to the stairs down. The stairs down are east. I already crossed the lava, so I should continue to the stairs down. I should move east."
    },
    {
        "task_id": "MiniHack-LavaCross-Levitate-Potion-Inv-v0",
        "state": [
            "You have a +1 club (weapon in hand)",
            "You have a +2 sling (alternate weapon; not wielded)",
            "You have 19 uncursed flint stones (in quiver pouch)",
            "You have 29 uncursed rocks",
            "You have an uncursed +0 leather armor (being worn)",
            "You have a yellow potion",
            "Strength: 18/18",
            "Dexterity: 15",
            "Constitution: 16",
            "Intelligence: 7",
            "Wisdom: 9",
            "Charisma: 10",
            "Depth: 1",
            "Gold: 0",
            "HP: 16/16",
            "Energy: 2/2",
            "AC: 8",
            "XP: 1/0",
            "Time: 1",
            "Position: 34|9",
            "Hunger: Not Hungry",
            "Monster Level: 0",
            "Encumbrance: Unencumbered",
            "Dungeon Number: 0",
            "Level Number: 1",
            "Score: 0",
            "Alignment: Neutral",
            "Condition: None",
            "You see a stairs down far east northeast",
            "You see a lava near east northeast and east",
            "You see a area of lava near east southeast",
            "You see a horizontal wall near southeast and south",
            "You see a horizontal wall very near north and northeast",
            "You see a vertical wall very near southwest and west",
            "You see a northwest room corner very near northwest",
            "f - a yellow potion."
        ],
        "admissible": ["move north", "move south", "move east", "move west", "move northwest", "move northeast", "move southwest", "move southeast", "drink a yellow potion"],
        "action": "drink a yellow potion",
        "act_explanation": "The first step of my task is to drink the potion. I have a yellow potion. I should drink the yellow potion."
    },
    {
        "task_id": "MiniHack-Room-Monster-5x5-v0",
        "state": [
            "You have a +0 short sword (weapon in hand)",
            "You have 14 +0 daggers (alternate weapon; not wielded)",
            "You have an uncursed +1 leather armor (being worn)",
            "You have an uncursed potion of sickness",
            "You have an uncursed lock pick",
            "You have an empty uncursed sack",
            "Strength: 15/15",
            "Dexterity: 15",
            "Constitution: 10",
            "Intelligence: 11",
            "Wisdom: 15",
            "Charisma: 9",
            "Depth: 1",
            "Gold: 0",
            "HP: 12/12",
            "Energy: 2/2",
            "AC: 7",
            "XP: 1/0",
            "Time: 2",
            "Position: 37|9",
            "Hunger: Not Hungry",
            "Monster Level: 0",
            "Encumbrance: Unencumbered",
            "Dungeon Number: 0",
            "Level Number: 1",
            "Score: 0",
            "Alignment: Chaotic",
            "Condition: None",
            "You see a stairs down near east",
            "You see a dark area near east, southeast, and south",
            "You see a dark area very near southwest and west",
            "You see a dark area adjacent north, northeast, and northwest",
            "You see a kobold adjacent south",
            "You hit the kobold."
        ],
        "admissible": ["move north", "move south", "move east", "move west", "move northwest", "move northeast", "move southwest", "move southeast", "attack the kobold"],
        "action": "attack the kobold",
        "act_explanation": "The first step of my task is to navigate to the stairs down while attacking monsters. The stairs down are east. I also see a kobold adjacent to me. A kobold is a monster. I can attack monsters adjacent to me. I should attack the kobold."
    },
    {
        "task_id": "MiniHack-Wear-v0",
        "state": [
            "You have a +1 club (weapon in hand)",
            "You have a +2 sling (alternate weapon; not wielded)",
            "You have 23 uncursed flint stones (in quiver pouch)",
            "You have 24 uncursed rocks",
            "You have an uncursed +0 leather armor (being worn)",
            "You have a robe",
            "Strength: 22/19",
            "Dexterity: 10",
            "Constitution: 17",
            "Intelligence: 10",
            "Wisdom: 8",
            "Charisma: 8",
            "Depth: 1",
            "Gold: 0",
            "HP: 16/16",
            "Energy: 2/2",
            "AC: 8",
            "XP: 1/0",
            "Time: 6",
            "Position: 40|11",
            "Hunger: Not Hungry",
            "Monster Level: 0",
            "Encumbrance: Unencumbered",
            "Dungeon Number: 0",
            "Level Number: 1",
            "Score: 0",
            "Alignment: Lawful",
            "Condition: None",
            "You see a dark area near north, south, southwest, west, and northwest",
            "You see a stairs up near west southwest",
            "You see a dark area adjacent northeast, east, and southeast",
            "f - a robe."
        ],
        "admissible": ["move north", "move south", "move east", "move west", "move northwest", "move northeast", "move southwest", "move southeast", "wear a robe"],
        "action": "wear a robe",
        "act_explanation": "The first step of my task is to pick up the robe. I already have a robe, so I've already picked it up. The description does not mention me wearing the robe that I have. I should wear the robe."
    },
]
