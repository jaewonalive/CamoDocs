import os

from openai import OpenAI
from .Model import Model


class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        if not 0 <= api_pos < len(api_keys):
            raise ValueError(
                f"api_key_use={api_pos} is out of range for {len(api_keys)} configured key(s)."
            )
        # The shipped config carries the "YOUR_API_KEY" placeholder, so fall
        # back to the environment rather than sending it to the API and
        # failing with a 401 on every call.
        api_key = api_keys[api_pos]
        if not api_key or api_key == "YOUR_API_KEY":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "No OpenAI API key: put one in the model config or export OPENAI_API_KEY."
                )
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = OpenAI(api_key=api_key)

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            )
            return completion.choices[0].message.content

        except Exception as e:
            # Do not swallow this. gen_adv.py --gen_answer stores the return
            # value as the query's gold answer, so returning "" on failure
            # would silently corrupt the dataset.
            raise RuntimeError(
                f"{self.name} query failed: {type(e).__name__}: {e}"
            ) from e