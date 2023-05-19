from typing import Dict, Union, List, Tuple
from itertools import chain
import numpy as np
from nle import nethack
from nle.nethack.actions import *
import minihack
from nle_language_wrapper import NLELanguageWrapper
from nle_language_wrapper.nle_language_obsv import NLELanguageObsv


NLE_LANG = NLELanguageObsv()


TASK_TO_DESC = {
    "MiniHack-Room-5x5-v0": "navigate to the stairs down.",
    "MiniHack-Room-15x15-v0": "navigate to the stairs down.",
    "MiniHack-Room-Random-5x5-v0": "navigate to the stairs down.",
    "MiniHack-Room-Random-15x15-v0": "navigate to the stairs down.",
    "MiniHack-Room-Monster-5x5-v0": "navigate to the stairs down while attacking monsters.",
    "MiniHack-Room-Monster-15x15-v0": "navigate to the stairs down and avoid monsters.",
    "MiniHack-Room-Trap-5x5-v0": "navigate to the stairs down and avoid traps.",
    "MiniHack-Room-Trap-15x15-v0": "navigate to the stairs down and avoid traps.",
    "MiniHack-Room-Dark-5x5-v0": "navigate to the stairs down in the dark.",
    "MiniHack-Room-Dark-15x15-v0": "navigate to the stairs down in the dark.",
    "MiniHack-Room-Ultimate-5x5-v0": "navigate to the stairs down in the dark while attacking monsters and avoiding traps.",
    "MiniHack-Room-Ultimate-15x15-v0": "navigate to the stairs down in the dark while avoiding monsters and traps.",
    "MiniHack-Eat-v0": "pick up and eat the apple.",
    "MiniHack-Wear-v0": "pick up and wear the robe.",
    "MiniHack-LavaCross-Levitate-Potion-Inv-v0": "drink the potion and navigate to the stairs down.",
    "MiniHack-LavaCross-Levitate-Ring-Inv-v0": "put on the ring and navigate to the stairs down.", 
    "MiniHack-LavaCross-v0": "pick up and use the item to cross the lava to the stairs down.", 
    "MiniHack-KeyRoom-S5-v0": "pick up and use the key to unlock the door and then navigate to the stairs down.",
    "MiniHack-KeyRoom-S15-v0": "use the key to unlock the door and navigate to the stairs down.",
    "MiniHack-KeyRoom-Dark-S5-v0": "use the key to unlock the door and navigate to the stairs down.",
    "MiniHack-KeyRoom-Dark-S15-v0": "use the key to unlock the door and navigate to the stairs down.",
    "MiniHack-WoD-Medium-v0": "pick up the wand and then navigate to the stairs down. When the monster is near, zap it.", 
    "MiniHack-Quest-Easy-v0": "zap the wand toward the lava and then navigate east toward the stairs down while attacking monsters.", 
    "MiniHack-Quest-Medium-v0": "follow to corridor to the giant rats, wait for them in the corridor to attack them one at a time, then zap the wand toward the lava to freeze it, and then navigate to the stairs down.", 
}


def get_message(obs) -> str:
    return NLE_LANG.text_message(obs["tty_chars"]).decode("latin-1")


def get_vision(obs) -> str:
    vision =  NLE_LANG.text_glyphs(obs["glyphs"], obs["blstats"]).decode("latin-1")
    for dir1 in ["east", "west", "north", "south"]:
        for dir2 in ["northwest", "northeast", "southwest", "southeast"]:
            vision = vision.replace(dir1 + dir2, dir1 + " " + dir2)
    return vision


def get_lang_obs(obs: Dict, as_list: bool = False) -> Union[str, List[str]]:    
    text_fields = {
        "text_glyphs": NLE_LANG.text_glyphs(obs["glyphs"], obs["blstats"]).decode(
            "latin-1"
        ),
        "text_message": NLE_LANG.text_message(obs["tty_chars"]).decode("latin-1"),
        "text_blstats": NLE_LANG.text_blstats(obs["blstats"]).decode("latin-1"),
        "text_inventory": NLE_LANG.text_inventory(
            obs["inv_strs"], obs["inv_letters"]
        ).decode("latin-1"),
        "text_cursor": NLE_LANG.text_cursor(
            obs["glyphs"], obs["blstats"], obs["tty_cursor"]
        ).decode("latin-1"),
    }

    for dir1 in ["east", "west", "north", "south"]:
        for dir2 in ["northwest", "northeast", "southwest", "southeast"]:
            text_fields["text_glyphs"] = text_fields["text_glyphs"].replace(dir1 + dir2, dir1 + " " + dir2)

    lang_obs = ["You have " + x[3:] for x in text_fields["text_inventory"].split("\n") if x] + \
        [x for x in text_fields["text_blstats"].split("\n") if x] + \
        ["You see a " + x for x in text_fields["text_glyphs"].split("\n") if x] + \
        ([text_fields["text_message"].replace("\n", "; ")] if text_fields["text_message"] else [])
    if as_list:
        return lang_obs
    else:
        return "\n".join(lang_obs)


def remove_item_parens(item):
    paren = item.find("(")
    if paren > -1:
        item = item[:paren-1]
    return item


def get_item_name(obs, char):
    if not isinstance(char, str):
        char = chr(char.value)
    for line in NLE_LANG.text_inventory(
                obs["inv_strs"], obs["inv_letters"]
            ).decode("latin-1").split("\n"):
        if len(line) > 3 and line[:3] == char + ": ":
            return remove_item_parens(line[3:])
    return ""


def get_inventory(obs):
    inv = obs["inv_strs"]
    return [x[x!=0].astype(np.uint8).tobytes().decode("utf-8").rstrip('\x00') for x in inv if any(x)]


def get_item_key(obs, item):
    inv = obs["inv_strs"]
    letters = obs["inv_letters"]
    for line, letter in zip(inv, letters):
        if item in line.astype(np.uint8).tobytes().decode("utf-8").rstrip('\x00'):
            return letter.astype(np.uint8).tobytes().decode("utf-8").rstrip('\x00')
    return None
    

def get_admissible(obs, allowed=ACTIONS) -> Tuple[List[str], List[List[str]]]:
    compass_actions = [
        "north",
        "south",
        "east",
        "west",
        "northwest",
        "northeast",
        "southwest",
        "southeast",
    ]
    lang_actions = ["move " + x for x in compass_actions]
    env_actions = compass_actions.copy()
    inv = get_inventory(obs)
    
    # Check for attack and apply actions
    for x in get_vision(obs).split("\n"):
        for m in range(nethack.NUMMONS):
            if nethack.permonst(m).mname + " adjacent" in x:
                lang_actions.append("attack the " + nethack.permonst(m).mname)
                env_actions.append(x.split()[-1])
        if "door adjacent" in x and any("key" in x.lower() for x in inv):
            lang_actions.append("use key")
            env_actions.append("a")

    # Check for pickup action
    message = get_message(obs)
    pickup_idx = message.find("You see here ")
    if pickup_idx > -1:
        lang_actions.append("pick up " + message[pickup_idx+13:].split(".")[0])
        env_actions.append(",")

    # Check for inventory acitons
    # TODO complete list of inventory actions
    for x in inv:
        if "wand" in x:
            key = get_item_key(obs, x)
            if key is not None:
                for direction in compass_actions:
                    lang_actions.append("zap " + get_item_name(obs, key) + " " + direction)
                    env_actions.append(["z", key, direction])
        elif any(y in x for y in ["apple", "pear", "banana"]):
            key = get_item_key(obs, x)
            if key is not None:
                lang_actions.append("eat " + get_item_name(obs, key))
                env_actions.append(["e", key])
        elif any(y in x for y in ["robe", "shoes", "boots"]) and "(being worn)" not in x:
            key = get_item_key(obs, x)
            if key is not None:
                lang_actions.append("wear " + get_item_name(obs, key))
                env_actions.append(["W", key])
        elif "potion" in x:
            key = get_item_key(obs, x)
            if key is not None:
                lang_actions.append("drink " + get_item_name(obs, key))
                env_actions.append(["q", key])
        elif "ring" in x and "(on right hand)" not in x and "(on left hand)" not in x:
            key = get_item_key(obs, x)
            if key is not None:
                lang_actions.append("put on " + get_item_name(obs, key))
                env_actions.append(["P", key, "r"])
        elif "horn" in x:
            key = get_item_key(obs, x)
            if key is not None:
                for direction in compass_actions:
                    lang_actions.append("blow horn " + direction)
                    env_actions.append(["a", key, "y", direction])

    env_actions = [[e] if isinstance(e, str) else e for e in env_actions]
    allowed = set(chain.from_iterable(l for e, l in NLELanguageWrapper.all_nle_action_map.items() if e in allowed))
    lang_actions, env_actions = zip(*((l, e) for l, e in zip(lang_actions, env_actions) if e[0] in allowed))
    return lang_actions, env_actions
