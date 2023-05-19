from typing import List, Union, Tuple
import torch
import random
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from actor import LLMActor


class LogitActor(LLMActor):
    def __init__(self, checkpoint="google/flan-t5-xl", temperature=.1, device="cuda", **kwargs):
        super().__init__(**kwargs)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            truncation_side="left",
            padding_size="right",
            pad_token_as_eos_token=False
        )
        self.temperature = temperature
        self.action_baselines = dict()

    def reset(self, task_description: str = ""):
        if task_description != self.task:
            self.action_baselines = dict()
        return super().reset(task_description)

    def get_actor_prompt(
            self,
            summary: str,
            task_description: str = ""
        ) -> str:
        prompt = self.prompt
        if task_description:
            prompt += " Your task is to " + task_description + " "
        prompt += " ".join([
            self.affordances, 
            summary + ".", 
            "\nYou choose to:"
        ])
        return prompt

    def get_score(
            self,
            state: Union[str, List[str]],
            actions: Union[str, List[str]], 
            task: str = None,
            baseline: float = 0,
            scale: float = 1
        ) -> torch.Tensor:
        if task is None:
            task = self.task
        if isinstance(actions, str):
            actions = [actions]
        if isinstance(state, list):
            state = ". ".join([x[:-1] if x[-1] == "." else x for x in state])

        with torch.no_grad():

            prompt_ids = self.tokenizer(
                self.get_actor_prompt(state, task), 
                return_tensors="pt"
            ).input_ids.to(self.model.device)
            encoder_cache = (
                self.model.encoder(prompt_ids, return_dict=True).last_hidden_state.repeat(len(actions), 1, 1),
            )

            action_inp = self.tokenizer(
                [self.tokenizer.pad_token + a for a in actions], 
                padding=True,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids.to(self.model.device)
            model_out = self.model(
                decoder_input_ids=action_inp[:, :-1],
                encoder_outputs=encoder_cache,
                return_dict=True
            )

            logits = torch.gather(model_out.logits, 2, action_inp[:, 1:].unsqueeze(-1)).squeeze(-1)

            # Get final (non padding) states
            score = []
            for i in range(action_inp.shape[0]):
                padding_idx = torch.where(action_inp[i, 1:] == self.tokenizer.pad_token_id)[0]
                if padding_idx.shape[0]:
                    score.append(torch.mean(logits[i, :padding_idx[0]], dim=0, keepdim=True))
                else:
                    score.append(torch.mean(logits[i], dim=0, keepdim=True))

            score = torch.cat(score)
            score = score * scale - baseline

        return score

    def get_action(
            self,
            lang_obs: Union[str, List[str]],
            lang_actions: List[str],
            env_actions: List[List[str]],
            return_tuple: bool = False
        ) -> Union[List[str], Tuple[List[str], str, str, int]]:

        # Get baseline scores for all actions
        for a in lang_actions:
            if a not in self.action_baselines:
                self.action_baselines[a] = self.get_score("", a, baseline=0).item()

        # Get scores for high actions
        baseline = torch.tensor([self.action_baselines[a] for a in lang_actions])
        scores = self.get_score(lang_obs, lang_actions, baseline=baseline.to(self.model.device))

        if torch.all(scores == -torch.inf):
            lang_action = random.choice(lang_actions)
        else:
            if self.temperature == 0:
                max_score = torch.max(scores)
                scores = torch.where(scores == max_score, 1, -torch.inf)
            else:
                scores /= self.temperature
            probs = torch.softmax(scores, 0)
            action_idx = torch.multinomial(probs, 1).item()
            lang_action = lang_actions[action_idx]

        if return_tuple:
            return env_actions[lang_actions.index(lang_action)], lang_action, "", 0
        else:
            return env_actions[lang_actions.index(lang_action)]
