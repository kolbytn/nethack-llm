from typing import Tuple, List
import os
import openai
import time
from openai.error import ServiceUnavailableError, RateLimitError, APIError, InvalidRequestError

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_chat(turns: List[str], max_len: int = 200, max_tries: int = 100, system_message: str = "") -> Tuple[str, int]:

    num_tries = 0
    while True:
        try:
            messages = [dict(role="system", content=system_message)] if system_message else []
            for i, content in enumerate(turns):
                messages.append(dict(role="user" if i % 2 == 0 else "assistant", content=content))
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=.7,
                max_tokens=max_len,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content, response.usage.total_tokens
        except ServiceUnavailableError as e:
            print("ServiceUnavailableError:", e)
            time.sleep(2)
        except RateLimitError as e:
            print("RateLimitError:", e)
            time.sleep(60)
        except APIError as e:
            print("APIError:", e)
            time.sleep(2)
        except InvalidRequestError as e:
            print("InvalidRequestError:", e)
            if len(turns) > 2:
                turns = turns[2:]
        num_tries += 1
        if num_tries >= max_tries:
            raise Exception()
