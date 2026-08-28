"""Adversarial-document loader for the evaluation stage.

CamoDocs generates its adversarial documents offline: `gen_synth_benign.py`
(Stage 1) -> `gen_adv.py` (Stage 2) -> `mix_and_create_adv_result.py`
(Stage 3, the HotFlip + coherence-filter optimisation). By the time
`evaluate.py` runs, the documents already exist on disk, so this module only
has to load them and pair them with their target queries.

The gradient-guided token replacement itself (`hotflip_attack`,
`GradientStorage`, `candidate_filter`, ...) lives in
`mix_and_create_adv_result.py`, which is where it is actually executed.
"""

from src.utils import load_json
from loguru import logger


class Attacker():
    def __init__(self, args, attack_method=None, adv_results_path=None,
                 adv_per_query=None, **kwargs) -> None:
        """`kwargs` absorbs the retriever handles (model, c_model, tokenizer,
        get_emb) that `evaluate.py` passes. They are unused here because the
        adversarial documents are pre-generated, and are accepted only so the
        call site does not have to special-case this attack."""
        self.args = args
        self.attack_method = attack_method if attack_method is not None else args.attack_method
        self.adv_per_query = adv_per_query if adv_per_query is not None else args.adv_per_query

        if adv_results_path is None:
            raise ValueError(
                "Attacker requires the path to the pre-generated adversarial "
                "documents. Pass --custom_attack_path pointing at a Stage-3 "
                "output file, e.g. "
                "data_examples/camodocs_hotpotqa_adv_text_merged.json"
            )
        self.all_adv_texts = load_json(adv_results_path)

        logger.info(f"Initializing attacker with method: {args.attack_method}")

    def get_attack(self, target_queries) -> list:
        '''
        This function returns adv_text_groups, which contains adv_texts for M queries
        For each query, if adv_per_query>1, we use different generated adv_texts or copies of the same adv_text
        '''
        adv_text_groups = []
        if self.attack_method == "LM_targeted":
            for i in range(len(target_queries)):
                question = target_queries[i]['query']
                id = target_queries[i]['id']

                if self.adv_per_query == -1:
                    adv_texts_b = self.all_adv_texts[id]['adv_texts']
                else:
                    adv_texts_b = self.all_adv_texts[id]['adv_texts'][:self.adv_per_query]

                # CamoDocs omits the query prefix; set --query_omitted False to
                # reproduce the PoisonedRAG baseline, which prepends it.
                if 'true' in self.args.query_omitted.lower():
                    adv_text_a = ""
                else:
                    adv_text_a = question + "."

                adv_texts = [adv_text_a + i for i in adv_texts_b]
                adv_text_groups.append(adv_texts)
        else:
            raise NotImplementedError
        return adv_text_groups
