from openai import OpenAI
import os
import time

class GPT():
    """OpenAI GPT victim wrapper with retry + failure tracking.

    Each query() call retries up to `max_retries` times with exponential
    backoff. If all retries fail, the call returns "" (preserving the
    previous contract) AND appends an entry to `self.failed_calls` so
    evaluate.py can surface the failures in a final summary
    instead of silently producing corrupted results.
    """

    def __init__(self, model_name, max_retries=5,
                 base_backoff=1.5, max_backoff=16.0):
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Export it before running: export OPENAI_API_KEY=<your-key>"
            )
        self.max_output_tokens = 1024
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        # GPT-5 series: requires max_completion_tokens, only supports temperature=1
        self.is_gpt5 = 'gpt-5' in model_name.lower()

        # Retry / failure-tracking state. Failures are NOT raised — the
        # caller still receives "" so the outer loop can continue. The
        # ledger lets evaluate.py distinguish "model legitimately
        # produced empty output" from "API failed after retries".
        # When a call fails the ledger records its qid, so failed
        # queries can be re-run later via --target_key_path.
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.failed_calls = []      # [{call_idx, qid, error}]
        # qid resolution: evaluate.py hands the client the list of N
        # qids and the period N (= number of unique queries in the run),
        # so qid_list[call_idx % qid_period] recovers the qid even for
        # multi-call defense helpers (conflict, divide_and_vote, etc.)
        # where each qid is visited several times in lock-step.
        self.qid_list = None
        self.qid_period = None
        self._call_idx = 0

    def _resolve_qid(self, call_idx):
        if self.qid_list is not None and self.qid_period:
            return self.qid_list[call_idx % self.qid_period]
        return None

    def query(self, msg, temperature=0.1):
        call_idx = self._call_idx
        self._call_idx += 1

        params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": msg},
            ],
        }
        if self.is_gpt5:
            params["max_completion_tokens"] = self.max_output_tokens
        else:
            params["temperature"] = temperature
            params["max_tokens"] = self.max_output_tokens

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(**params)
                return completion.choices[0].message.content
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    delay = min(self.base_backoff * (2 ** (attempt - 1)),
                                self.max_backoff)
                    print(f"[GPT] attempt {attempt}/{self.max_retries} "
                          f"failed at call_idx={call_idx}: "
                          f"{type(e).__name__}: {e} — sleeping {delay:.1f}s")
                    time.sleep(delay)

        # All retries exhausted. Resolve the qid for re-run via
        # --target_key_path (qid_list[call_idx % qid_period]).
        self.failed_calls.append({
            "call_idx": call_idx,
            "qid": self._resolve_qid(call_idx),
            "error": f"{type(last_err).__name__}: {last_err}",
        })
        print(f"[GPT] !!! ALL {self.max_retries} RETRIES FAILED "
              f"at call_idx={call_idx}: {last_err}")
        return ""
