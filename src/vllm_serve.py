from openai import OpenAI


class VLLMServeLLM:
    """
    Client for a locally-hosted vLLM OpenAI-compatible server.

    Used by the LLM Filter defense as the critic backend; see
    scripts/gpt_oss_safeguard_vllm_serve.sh to start the server and
    scripts/llm_filter_eval.sh to run the evaluation.

    Usage:
        # Start the server first (this is what the shipped script does):
        #   vllm serve openai/gpt-oss-safeguard-20b \
        #       --tensor-parallel-size 4 --port 8001
        llm = VLLMServeLLM(
            model_name="openai/gpt-oss-safeguard-20b",
            base_url="http://localhost:8001/v1",
        )
    """

    def __init__(self, model_name, base_url="http://localhost:8000/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",  # vLLM serve does not require a real key
        )
        self.model_name = model_name
        self.max_output_tokens = 4096

    def query(self, msg, temperature=0.1):
        """
        Standard query -- returns the response text.

        gpt-oss-safeguard emits its deliberation in a separate
        `reasoning_content` field rather than inline <think> tags. It is
        re-wrapped in <think>...</think> here so that a single
        `strip_thinking_tags` call downstream removes it regardless of
        which form the server used.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg},
                ],
            )
            msg_obj = completion.choices[0].message
            content = msg_obj.content or ""
            reasoning = getattr(msg_obj, "reasoning_content", None) or ""
            if reasoning:
                response = f"<think>\n{reasoning}\n</think>\n{content}"
            else:
                response = content
        except Exception as e:
            print(f"[VLLMServeLLM] Error: {e}")
            response = ""
        return response
