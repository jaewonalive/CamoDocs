from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.utils import load_beir_datasets, save_json


_SYSTEM_MSG = "You are a helpful assistant."


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def _build_prompt(question, correct_answer, adv_per_query, dataset):
    """Build the Stage-1 prompt asking the synthesizer for `adv_per_query`
    benign passages that support the correct answer without stating it."""
    gen_adv_prompt = (
    f"This is my query: {question}. "
    f"Please craft {adv_per_query} concise, relevant corpora "
        "that help answer the query. "
        "The corpora must not explicitly state the answer (e.g., "
        "'the answer to the query is ...'). "
        "They should read like retrieved documents in a "
        "retrieval-augmented generation (RAG) system, so that an "
        "LLM prompted with the query can produce the correct "
        "answer. Each corpus should be approximately 100 words "
        "long. Return your reply as a JSON-formatted string."
    )

    for k in range(adv_per_query):
        if k == adv_per_query - 1:
            gen_adv_prompt += f"corpus{k+1}."
        else:
            gen_adv_prompt += f"corpus{k+1}, "
    return gen_adv_prompt


def _parse_corpora(raw_response, adv_per_query):
    """Returns a list of N corpus strings, or None on parse failure."""
    try:
        adv_corpus = json.loads(raw_response)
    except Exception:
        return None
    if not isinstance(adv_corpus, dict):
        return None
    out = []
    for k in range(adv_per_query):
        key = f"corpus{k+1}"
        if key not in adv_corpus:
            return None
        t = str(adv_corpus[key])
        if t.startswith('"'):
            t = t[1:]
        if t.endswith('"'):
            t = t[:-1]
        out.append(t)
    return out


async def _gen_one(
    client,
    sem: asyncio.Semaphore,
    *,
    model: str,
    question: str,
    correct_answer: str,
    adv_per_query: int,
    dataset: str,
    max_retries: int = 30,
    base_backoff: float = 1.5,
    max_backoff: float = 16.0,
    logger=None,
):
    """One async generation attempt with retry on transient + parse failures.

    Returns list[str] of length `adv_per_query`, or None if every retry
    fails. The caller (the orchestrator) records a None as a failed qid
    rather than crashing the whole batch.
    """
    prompt = _build_prompt(question, correct_answer, adv_per_query, dataset)
    delay = base_backoff
    async with sem:
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=1.0,
                    messages=[
                        {"role": "system", "content": _SYSTEM_MSG},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or ""
            except Exception as e:
                if logger:
                    logger.warning(
                        f"[gen_synth] api error attempt {attempt}/{max_retries}: "
                        f"{type(e).__name__}: {e}"
                    )
                if attempt < max_retries:
                    await asyncio.sleep(min(delay, max_backoff))
                    delay = min(delay * 2, max_backoff)
                    continue
                return None

            corpora = _parse_corpora(raw, adv_per_query)
            if corpora is not None:
                return corpora
            # Parse miss — re-prompt with a small backoff. The model
            # occasionally drops a corpus key under temperature=1.0.
            if logger:
                logger.warning(
                    f"[gen_synth] parse miss attempt {attempt}/{max_retries}  "
                    f"raw_head={raw[:120]!r}"
                )
            if attempt < max_retries:
                await asyncio.sleep(min(base_backoff, max_backoff))
    return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Dataset + selection
    p.add_argument("--eval_dataset", type=str, default="hotpotqa",
                   choices=["hotpotqa", "nq", "msmarco"],
                   help="BEIR dataset the target queries come from.")
    p.add_argument("--split", type=str, default="test",
                   help="BEIR split. MS-MARCO uses 'train'; HotpotQA and NQ use 'test'.")
    p.add_argument("--queries_jsonl_path", type=str,
                   help="BEIR queries.jsonl. HotpotQA only: used to pull "
                        "metadata.answer when --target_queries_path is a bare "
                        "list. NQ and MS-MARCO have empty query metadata in "
                        "BEIR, so their answers must come from the target file "
                        "or --query_answer_path.")
    p.add_argument("--target_queries_path", type=str, required=True,
                   help="JSON list (or dict keyed by id) of query ids to process. "
                        "A dict may also carry per-query 'correct answer' fields.")
    p.add_argument("--query_answer_path", type=str, default=None,
                   help="MS-MARCO answers (qid -> {'answer': ...}), joined from "
                        "the ms_marco v2.1 QnA set. Required for MS-MARCO unless "
                        "--target_queries_path already carries 'correct answer'.")
    p.add_argument("--save_path", type=str,
                   default="results/synth_benign_results",
                   help="Output directory.")
    p.add_argument("--file_name", type=str, required=True,
                   help="Output base name (no .json). The final file is "
                        "<save_path>/<file_name>.json; checkpoints land "
                        "at <save_path>/<file_name>_cnt_<n>.json.")
    p.add_argument("--start_idx", type=int, default=None,
                   help="Resume from the checkpoint <file_name>_cnt_<N>.json. "
                        "Queries already present in that file are skipped; "
                        "everything else, including previous failures, is retried.")
    p.add_argument("--adv_per_query", type=int, default=5,
                   help="Synthetic benign passages to generate per query (carriers for Stage 3).")
    # Model + async
    p.add_argument("--model_name", type=str,
                   default="gpt-4o-mini-2024-07-18",
                   help="OpenAI chat model used as the synthesizer.")
    p.add_argument("--api_key", type=str,
                   default=os.environ.get("OPENAI_API_KEY", ""),
                   help="OpenAI API key. Required; defaults to $OPENAI_API_KEY.")
    p.add_argument("--concurrency", type=int, default=50,
                   help="Max concurrent in-flight requests. 50 is safe on "
                        "Tier 5 OpenAI for gpt-4o-mini.")
    p.add_argument("--checkpoint_every", type=int, default=100,
                   help="Dump a checkpoint JSON every N successful queries.")
    return p.parse_args()


def _load_target_queries_with_answers(path):
    """Try to use --target_queries_path as the ONLY source of qid + correct
    answer when it's a rich dict (e.g., poisonedrag's gen_adv output:
    {qid: {"id", "question", "correct answer", "incorrect answer",
    "adv_texts"}}).

    Returns (qids, answers_dict_or_None). If answers can't be extracted
    (path is a list of plain qids, or dict values lack 'correct answer'),
    returns (qids, None) and the caller must fall back to queries.jsonl
    or --query_answer_path.
    """
    with open(path) as f:
        td = json.load(f)
    if isinstance(td, list):
        return list(td), None
    if not isinstance(td, dict):
        raise SystemExit(f"--target_queries_path must be list or dict, got {type(td)}")
    qids = list(td.keys())
    # Detect the rich format by sampling one entry.
    first = td[qids[0]]
    if (isinstance(first, dict)
            and "correct answer" in first):
        answers = {qid: td[qid]["correct answer"] for qid in qids}
        return qids, answers
    return qids, None


def _load_dataset_and_queries(args, logger):
    """Returns (queries, queries_answer, queries_incorrect_or_None).

    Three supported shapes for --target_queries_path:
      (a) JSON list of qids — needs a separate answer source:
          --queries_jsonl_path for HotpotQA, --query_answer_path for
          MS-MARCO. NQ has no such fallback (BEIR carries no answers),
          so NQ requires shape (b).
      (b) JSON dict {qid: {...}} where each value has 'correct answer' —
          self-contained, no queries.jsonl needed.
    """
    corpus, queries, qrels = load_beir_datasets(args.eval_dataset, args.split)

    logger.info(f"Total BEIR queries: {len(queries)}")

    queries_incorrect = None

    if args.eval_dataset == "hotpotqa":
        sel, queries_answer_from_target = _load_target_queries_with_answers(
            args.target_queries_path
        )
        queries = {qid: queries[qid] for qid in sel if qid in queries}

        if queries_answer_from_target is not None:
            queries_answer = queries_answer_from_target
            logger.info(
                "[load] using 'correct answer' fields from --target_queries_path "
                "(self-contained mode)"
            )
        else:
            assert args.queries_jsonl_path, (
                "hotpotqa: --target_queries_path is a list (no answers); "
                "you must also pass --queries_jsonl_path"
            )
            queries_answer = {}
            with open(args.queries_jsonl_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        qd = json.loads(line)
                        queries_answer[qd["_id"]] = qd["metadata"]["answer"]
            for qid in queries:
                if qid not in queries_answer:
                    raise SystemExit(f"{qid} missing from queries.jsonl")

    elif args.eval_dataset == "nq":
        sel, queries_answer_from_target = _load_target_queries_with_answers(
            args.target_queries_path
        )
        queries = {qid: queries[qid] for qid in sel if qid in queries}
        if queries_answer_from_target is not None:
            queries_answer = queries_answer_from_target
            logger.info(
                "[load] using 'correct answer' fields from --target_queries_path"
            )
        else:
            # BEIR's NQ port has empty query metadata, so queries.jsonl cannot
            # supply answers. As described in Appendix A.1 of the paper, NQ
            # answers were joined from the DPR-preprocessed data. The target
            # file must carry them.
            raise SystemExit(
                "nq: --target_queries_path is a plain list of ids, but BEIR's NQ "
                "port carries no answers in query metadata. Pass a dict of "
                "{qid: {'correct answer': ...}} instead."
            )

    elif args.eval_dataset == "msmarco":
        sel, queries_answer_from_target = _load_target_queries_with_answers(
            args.target_queries_path
        )

        if queries_answer_from_target is not None:
            queries_answer = queries_answer_from_target
            queries = {qid: queries[qid] for qid in sel if qid in queries}
            logger.info(
                "[load] using 'correct answer' fields from --target_queries_path"
            )
        else:
            # BEIR's MS-MARCO port has empty query metadata (0 of ~510k queries
            # carry metadata.answer), so answers must come from
            # --query_answer_path. As described in Appendix A.1 of the paper,
            # those answers were joined from the ms_marco v2.1 QnA set.
            if args.query_answer_path is None:
                raise SystemExit(
                    "msmarco: --target_queries_path is a plain list of ids, so it "
                    "carries no answers, and BEIR's MS-MARCO port has none in query "
                    "metadata. Either pass a dict of {qid: {'correct answer': ...}} "
                    "as --target_queries_path, or add --query_answer_path."
                )
            with open(args.query_answer_path) as f:
                qad = json.load(f)
            queries_answer = {k: v["answer"] for k, v in qad.items()}
            queries = {qid: queries[qid] for qid in sel if qid in queries}

    return queries, queries_answer, queries_incorrect


async def _run_async(args, logger):
    from openai import AsyncOpenAI

    os.makedirs(args.save_path, exist_ok=True)

    queries, queries_answer, _ = _load_dataset_and_queries(args, logger)
    qid_list = list(queries.keys())
    logger.info(f"Selected queries: {len(qid_list)}")

    # Resume support — load the checkpoint named by --start_idx.
    results = {}
    if args.start_idx is not None:
        ckpt = os.path.join(args.save_path, f"{args.file_name}_cnt_{args.start_idx}.json")
        with open(ckpt) as f:
            results = json.load(f)
        # Resume by identity, not by position. Requests run concurrently, so a
        # checkpoint written at N completions holds an arbitrary N-subset of
        # qid_list -- not its first N entries. Slicing by index would silently
        # drop queries that had not finished yet. Filtering on the checkpoint's
        # keys also retries anything that failed.
        before = len(qid_list)
        qid_list = [q for q in qid_list if q not in results]
        logger.info(f"Resuming from {ckpt}: {len(results)} already done, "
                    f"{len(qid_list)} of {before} remaining.")

    if not args.api_key:
        raise SystemExit("Set --api_key or $OPENAI_API_KEY.")
    client = AsyncOpenAI(api_key=args.api_key, max_retries=5)

    sem = asyncio.Semaphore(args.concurrency)
    ckpt_lock = asyncio.Lock()
    n_done = len(results)
    n_failed = 0

    async def _one(qid):
        nonlocal n_done, n_failed
        question = queries[qid]
        correct = queries_answer[qid]
        corpora = await _gen_one(
            client, sem,
            model=args.model_name,
            question=question,
            correct_answer=correct,
            adv_per_query=args.adv_per_query,
            dataset=args.eval_dataset,
            logger=logger,
        )
        async with ckpt_lock:
            if corpora is None:
                n_failed += 1
                logger.warning(
                    f"[gen_synth] FAIL qid={qid} after retries — skipping"
                )
                return
            results[qid] = {
                "id": qid,
                "question": question,
                "correct answer": correct,
                "synth_texts": corpora,
            }
            n_done += 1
            if n_done % args.checkpoint_every == 0:
                ckpt = os.path.join(
                    args.save_path, f"{args.file_name}_cnt_{n_done}.json"
                )
                save_json(results, ckpt)
                logger.info(
                    f"[gen_synth] checkpoint {n_done}/{len(queries)} -> {ckpt}  "
                    f"(failed so far: {n_failed})"
                )

    logger.info(
        f"[gen_synth] launching async batch: "
        f"model={args.model_name}  concurrency={args.concurrency}  "
        f"adv_per_query={args.adv_per_query}"
    )
    await asyncio.gather(*[_one(q) for q in qid_list])

    final = os.path.join(args.save_path, f"{args.file_name}.json")
    save_json(results, final)
    logger.info(
        f"[gen_synth] done. wrote {len(results)} entries -> {final}  "
        f"(failed: {n_failed})"
    )
    if not results:
        raise SystemExit(
            f"[gen_synth] no queries succeeded ({n_failed} failed); {final} is empty."
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    args = parse_args()
    logger.info(f"args: {vars(args)}")
    asyncio.run(_run_async(args, logger))


if __name__ == "__main__":
    main()
