# LLM Actor for NetHack

Repository for running LLMs on NetHack tasks. Contains functionality for using Seq2Seq models as well as openai's gpt3-turbo model via the chat api.

## Installation

```
pip install -r requirements.txt
```

## Usage

To run Huggingface Seq2Seq actor:

```
python rollout.py --exp_name my_t5_test --actor google/flan-t5-xl --task MiniHack-Room-5x5-v0
```

To run GPT3 actor w/ fewshot explanations:

```
OPENAI_API_KEY="your-key-here" python rollout.py --exp_name my_gpt_test --actor gpt --cot --task MiniHack-Room-5x5-v0
```
