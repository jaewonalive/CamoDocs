# Answer-condensation prompt used by gen_adv.py --gen_answer (Appendix A.1).
# MS-MARCO's gold answers were produced with this prompt via gpt-4o-mini.
#
# NOTE: the RAG prompt used at evaluation time is a DIFFERENT string and lives
# in src/prompts.py. Do not confuse the two.
CONCISE_PROMPT = (
    "You are a helpful assistant. Below is a user query, relevant context, and a ground-truth answer. "
    "Answer the question using the provided context. Keep the answer short and concise—"
    "ideally a single word or a few words that capture the main point. "
    "If you cannot find the answer in the context, reply with \"I don't know\".\n\n"
    "Context: [context]\n\n"
    "Query: [question]\n\n"
    "Ground-truth answer: [answer]\n\n"
    "Your answer:"
)


def wrap_prompt(question, context, prompt_id=5, correct_answer=None) -> str:
    """Fill CONCISE_PROMPT. `context` is the list of gold contexts for the query.

    prompt_id exists only to keep gen_adv.py's call signature unchanged; 5 is
    the sole supported value.
    """
    if prompt_id != 5:
        raise ValueError(
            f"synth_src.prompts.wrap_prompt supports prompt_id=5 only, got {prompt_id}. "
            "The evaluation-time RAG prompt lives in src/prompts.py."
        )
    if correct_answer is None:
        raise ValueError("wrap_prompt(prompt_id=5) requires correct_answer, got None.")

    if not isinstance(context, list):
        raise TypeError(f"context must be a list of gold contexts, got {type(context).__name__}.")
    context_str = "\n".join(context)
    return (CONCISE_PROMPT
            .replace('[question]', question)
            .replace('[context]', context_str)
            .replace('[answer]', correct_answer))
