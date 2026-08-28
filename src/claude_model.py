import os
import time
import threading
import anthropic


class ClaudeLLM:
    """
    Anthropic Claude API client with retry + failure tracking.
    Requires ANTHROPIC_API_KEY environment variable.

    Each query() call retries up to `max_retries` times with exponential
    backoff. If all retries fail, the call returns "" (preserving the
    previous contract) AND appends an entry to `self.failed_calls` so
    evaluate.py can surface the failures in a final summary instead
    of silently producing corrupted results.

    Usage:
        llm = ClaudeLLM("claude-haiku-4-5")
        response = llm.query("What is 2+2?")
    """

    def __init__(self, model_name, max_retries=5,
                 base_backoff=1.5, max_backoff=16.0):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it before running: export ANTHROPIC_API_KEY=<your-key>"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        # Match the GPT victim's 1024-token budget so the two victims
        # produce comparable-length completions on identical prompts.
        # HotpotQA-style short-answer queries fit comfortably; reasoning
        # models that emit visible chain-of-thought may need more.
        self.max_output_tokens = 1024

        # Retry / failure tracking. The ledger and the call counter are
        # guarded by a lock so the client stays safe if a caller ever
        # drives query() from several threads. When a call fails the
        # ledger records its qid, so failed queries can be re-run via
        # --target_key_path.
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.failed_calls = []     # [{call_idx, qid, error}]
        # qid resolution: evaluate.py hands the client the list of N
        # qids and the period N (= number of unique queries in the run),
        # so qid_list[call_idx % qid_period] recovers the qid even for
        # multi-call defense helpers (conflict, divide_and_vote, etc.)
        # where each qid is visited several times in lock-step. This
        # mapping assumes calls are issued sequentially, which is how
        # evaluate.py drives it.
        self.qid_list = None
        self.qid_period = None
        self._call_counter = 0
        self._lock = threading.Lock()

    def _resolve_qid(self, call_idx):
        if self.qid_list is not None and self.qid_period:
            return self.qid_list[call_idx % self.qid_period]
        return None

    def _next_idx(self) -> int:
        with self._lock:
            i = self._call_counter
            self._call_counter += 1
            return i

    def query(self, msg, temperature=0.1):
        """Standard query — returns the final answer text. Empty string
        means EITHER the model legitimately produced empty output OR all
        retries failed (check self.failed_calls to disambiguate)."""
        call_idx = self._next_idx()

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_output_tokens,
                    temperature=temperature,
                    # Match the GPT victim's system message so the two
                    # victims share the same prompt template. Anthropic's
                    # API has a dedicated `system` parameter (separate
                    # from `messages`) for this purpose.
                    system="You are a helpful assistant.",
                    messages=[{"role": "user", "content": msg}],
                )
                return response.content[0].text
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    delay = min(self.base_backoff * (2 ** (attempt - 1)),
                                self.max_backoff)
                    print(f"[ClaudeLLM] attempt {attempt}/{self.max_retries} "
                          f"failed at call_idx={call_idx}: "
                          f"{type(e).__name__}: {e} — sleeping {delay:.1f}s")
                    time.sleep(delay)

        # All retries exhausted. Resolve the qid for re-run via
        # --target_key_path (qid_list[call_idx % qid_period]).
        with self._lock:
            self.failed_calls.append({
                "call_idx": call_idx,
                "qid": self._resolve_qid(call_idx),
                "error": f"{type(last_err).__name__}: {last_err}",
            })
        print(f"[ClaudeLLM] !!! ALL {self.max_retries} RETRIES FAILED "
              f"at call_idx={call_idx}: {last_err}")
        return ""
