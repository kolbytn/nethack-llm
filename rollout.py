import json
from tqdm import tqdm
from argparse import ArgumentParser

from actor.random_actor import RandomActor
from actor.chat_actor import ChatActor
from actor.logit_actor import LogitActor
from envs.lang_env import LangEnv
from utils.nle_utils import TASK_TO_DESC


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate rollout data")
    parser.add_argument("--exp_name", type=str, default="test", help="File name for saves")
    parser.add_argument("--task", type=str, default="", help="Task to evaluate on, default is all tasks")
    parser.add_argument("--actor", type=str, default="random", help="Can be random, gpt, or a path to a seq2seq huggingface model")
    parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts to evaluate")
    parser.add_argument("--max_episode_steps", type=int, default=None, help="Max episode steps")
    parser.add_argument("--fewshot", type=int, default=4, help="How many fewshot examples to use for gpt")
    parser.add_argument("--action_temp", type=float, default=1, help="Sampling temperature for action policy")
    parser.add_argument("--cot", action="store_true", help="Use explanaitons for actor")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"

    if args.actor == "random":
        actor = RandomActor()
    elif args.actor == "gpt":
        actor = ChatActor(fewshot=args.fewshot, use_cot=args.cot)
    else:
        actor = LogitActor(args.actor, temperature=args.action_temp)

    if args.task:
        tasks = [args.task]
    else:
        tasks = TASK_TO_DESC.keys()

    results = {
        x: dict(reward=0, success=0, death=0) 
        for x in tasks
    }
    
    for task in tasks:
        env = LangEnv(task)

        print("Starting Task:", task)
        pbar = tqdm(range(args.num_rollouts))
        for rollout_id in range(args.num_rollouts):

            lang_obs_list = env.reset()
            description = env.get_task()

            actor.reset(description)
            cum_reward = 0
            steps = 0
            done = False
            while not done:

                lang_actions, env_actions = env.get_actions()

                env_action = actor.get_action(
                    lang_obs_list, 
                    lang_actions, 
                    env_actions, 
                    return_tuple=False
                )

                if not isinstance(env_action, list):
                    env_action = [env_action]
                for a in env_action:
                    lang_obs_list, reward, done, info = env.step(a)
                    cum_reward += reward
                    steps += 1
                    if done:
                        break

                if args.max_episode_steps is not None and steps >= args.max_episode_steps:
                    done = True

            results[task]["reward"] += cum_reward / args.num_rollouts
            if reward > 0:
                results[task]["success"] += 1 / args.num_rollouts
            elif "end_status" in info and info["end_status"] == 1:
                results[task]["death"] += 1 / args.num_rollouts
            pbar.update(1)
            pbar.set_description("Successes {}/{}".format(int(results[task]["success"] * args.num_rollouts), rollout_id + 1))

        with open(args.exp_name + ".json", "w") as f:
            json.dump(results, f, indent=4)
