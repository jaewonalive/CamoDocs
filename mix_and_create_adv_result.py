import argparse
import json
import sys
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")



from sentence_transformers import SentenceTransformer
import torch
import random


from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import XLMRobertaForMaskedLM

import os

import time
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


def get_embeddings(model):
    """Returns the wordpiece embedding module."""

    # This can be different for different models; the following is tested for Contriever
    if isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings

    elif isinstance(model, XLMRobertaForMaskedLM):
        embeddings = model.roberta.embeddings.word_embeddings

    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None,
                   banned_ids=None):
    """Returns the top candidate replacements."""

    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1

        if banned_ids:
            banned = torch.as_tensor(banned_ids, device=gradient_dot_embedding_matrix.device, dtype=torch.long)
            gradient_dot_embedding_matrix.index_fill_(0, banned, float("-inf"))

        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


def split_into_sentences_nltk(text: str) -> list[str]:
    return [s.strip() for s in sent_tokenize(text) if s.strip()]



def join_splitted_text(text: list, break_num: int ) -> list[str]:
    

    if break_num == -1:
        return text

    else:
        
        total_sent_num = len(text)
        each_sent_num = total_sent_num // break_num

        start_idx = 0

        joined_list = []

        if each_sent_num == 0 :
            break_num = total_sent_num
            each_sent_num = 1

        for idx in range(break_num):

            if idx == break_num - 1:
                cur_target_sent = text[start_idx : ]

            else:
                
                cur_target_sent = text[start_idx : start_idx + each_sent_num]
                start_idx += each_sent_num


            concat_result = ''.join(cur_target_sent)

            joined_list.append(concat_result)

        return joined_list



def split_docs(benign_topk_docs_list, target_adv_texts, break_num) :
    joined_text_list = []
    joined_benign_text_list = []


    prev_joined_benign_text = [" "]
    for adv_text_idx in range(len(target_adv_texts)):

        one_text = target_adv_texts[adv_text_idx]
        splitted_text = split_into_sentences_nltk(one_text)
        joined_text = join_splitted_text(text = splitted_text, break_num = break_num)
        joined_text_len = len(joined_text)


        one_benign_text = benign_topk_docs_list[adv_text_idx]
        splitted_benign_text = split_into_sentences_nltk(one_benign_text)
        joined_benign_text = join_splitted_text(text = splitted_benign_text, break_num = break_num)

        if len(joined_benign_text) == 0 :
            print("empty joined benign text is detected. Reuse previous joined benign text.")
            joined_benign_text = prev_joined_benign_text


        prev_joined_benign_text = joined_benign_text
        
        
        joined_benign_text_len = len(joined_benign_text)

        if joined_text_len < break_num :
            diff_num = break_num - joined_text_len
            adv_supple_list = [num for num in range(joined_text_len)]
            adv_supple_list = adv_supple_list * diff_num # for indexing and we use round-robin approach.

            for add_idx in range(diff_num):
                joined_text.append(joined_text[adv_supple_list[add_idx]])

            joined_text_len = len(joined_text)


        if joined_text_len > joined_benign_text_len:
            
            diff_num = joined_text_len - joined_benign_text_len
            

            # Round-robin pad: guards against an empty benign list.
            benign_supple_list = [num for num in range(joined_benign_text_len)]
            
            benign_supple_list = benign_supple_list * diff_num


            for add_idx in range(diff_num):
                joined_benign_text.append(joined_benign_text[benign_supple_list[add_idx]]) # 

        for idx in range(break_num):
            joined_text_list.append(joined_text[idx])
            joined_benign_text_list.append(joined_benign_text[idx])


    return joined_benign_text_list, joined_text_list




def pick_safe_token_to_flip(adv_inputs, j, tokenizer, prevention_mask=None):
    """
    Return a *Python int* position in adv_inputs["input_ids"][j] that is safe to flip
    (not CLS/SEP/PAD/MASK/BOS/EOS and with attention_mask==1). Returns None if none.
    """
    # current ids/mask for example j (CPU for easy indexing)
    ids_j  = adv_inputs["input_ids"][j].detach().cpu()          # [seq_len]
    attn_j = adv_inputs["attention_mask"][j].detach().cpu().bool()

    if prevention_mask is not None:
        prevention_mask = prevention_mask.detach().cpu().bool()
    else:
        prevention_mask = torch.zeros_like(ids_j, dtype=torch.bool)

    # collect special token ids from tokenizer (skip Nones)
    names = ("cls_token_id","sep_token_id","pad_token_id",
            "mask_token_id","bos_token_id","eos_token_id")
    specials = [getattr(tokenizer, n, None) for n in names]
    specials = [t for t in specials if isinstance(t, int)]

    # mask out specials (and padding via attn_j)
    special_mask = torch.zeros_like(ids_j, dtype=torch.bool)
    for t in specials:
        special_mask |= (ids_j == t)

    allowed_mask = attn_j & ~special_mask & ~prevention_mask
    allowed_idx  = torch.nonzero(allowed_mask, as_tuple=False).squeeze(1)  # e.g., tensor([3, 8, 12])

    # convert to python list and choose uniformly
    pos_list = allowed_idx.tolist()
    print("allowed idx : ", pos_list)
    if not pos_list:
        return None
    return int(random.choice(pos_list)), specials


def compute_perplexity(input_ids, model, device):
    """
    Calculate the perplexity of the input_ids using the model.
    """

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)


def candidate_filter(candidates,
            num_candidates=1,
            token_to_flip=None,
            adv_passage_ids=None,
            ppl_model=None,
            cluster_embedding_tokenizer=None,
            ppl_tokenizer=None):
    """Return the `num_candidates` candidates with the LOWEST perplexity.

    Perplexity is negated before `topk`, so `topk` selects minima -- i.e. the
    most fluent replacements under `ppl_model`. This is the coherence filter
    (C_1 -> C_2 in the paper).
    """
    device='cuda'

    with torch.no_grad():
    
        ppl_scores = []
        temp_adv_passage = adv_passage_ids.clone()
        for candidate in candidates:
            temp_adv_passage[token_to_flip] = candidate

            # detoken
            detok_passage = cluster_embedding_tokenizer.decode(temp_adv_passage.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # retokenizer with ppl_tokenizer

            retok_input_ids = ppl_tokenizer(detok_passage, return_tensors="pt", truncation=False, padding=False)['input_ids'].to(device)


            ppl_score = compute_perplexity(retok_input_ids, ppl_model, device) * -1
            ppl_scores.append(ppl_score)
   

        ppl_scores = torch.tensor(ppl_scores)
        _, top_k_ids = ppl_scores.topk(num_candidates)
        candidates = candidates[top_k_ids]

    return candidates



parser = argparse.ArgumentParser(description='test')

parser.add_argument('--custom_attack_path', type=str, required=True,
                    help='Stage-2 output: LLM-drafted adversarial answers per target query.')
parser.add_argument('--benign_score_path', type=str, required=True,
                    help='Cached BEIR retrieval scores, e.g. '
                         'results/beir_results/hotpotqa-contriever.json.')
# -1 indicates splitting the text by every sentence.
parser.add_argument('--result_path', type=str, required=True,
                    help='Output directory for the Stage-3 result JSON.')
parser.add_argument('--file_name', type=str, required=True,
                    help='Base name of the output JSON (shard suffix added when sharding).')
parser.add_argument('--eval_dataset', default="hotpotqa", type=str,
                    help="BEIR dataset the target queries come from.")
parser.add_argument('--split', default="test", type=str,
                    help="BEIR split. MS-MARCO uses 'train'; HotpotQA and NQ use 'test'.")
parser.add_argument('--num_cand', type=int, default=1000,
                    help="Candidate tokens per HotFlip step (m = 1000 in the paper).")
parser.add_argument('--num_iter', type=int, default=30,
                    help="Token-replacement iterations per document (alpha = 30 in the paper).")
parser.add_argument('--adv_per_query', type=int, default=5,
                    help="Adversarial documents per query before chunking.")
parser.add_argument('--debug_mode', type=str2bool, default=False,
                    help="Verbose per-iteration logging of token flips.")
parser.add_argument('--break_num', type=int, default=None, help="how many pieces you are going to create with one passage.")
parser.add_argument('--manual_world_size', type=int, default=-1,
                    help="Shard count for splitting the query list across parallel jobs (-1 = no sharding).")
parser.add_argument('--manual_rank', type=int, default=-1,
                    help="Shard index in [0, manual_world_size). Output file name embeds it; merge with merge_shards.py.")
parser.add_argument('--data_num', type=int, default=1000,
                    help="Keep only the first N target queries (slice applied before sharding).")
parser.add_argument('--synth_docs_path', type=str, required=True,
                    help='Stage-1 output holding the synthetic benign documents used as carriers.')
parser.add_argument('--coherence_filter', type=str2bool, default=False,
                    help="Enable the GPT-2 coherence filter over HotFlip candidates (C_1 -> C_2 in the paper).")
parser.add_argument('--num_coherence_cand', type=int, default=100,
                    help="Candidates kept by the GPT-2 coherence filter (m' = 100 in the paper).")

parser.add_argument('--surrogate_model_name', type=str,
                    default='sentence-transformers/msmarco-roberta-base-ance-firstp',
                    help="Surrogate dense encoder for the HotFlip optimization. "
                         "Default ANCE matches the original CamoDocs paper (strict black-box). "
                         "Try 'sentence-transformers/all-mpnet-base-v2' for a wider-scale black-box "
                         "or 'princeton-nlp/sup-simcse-bert-base-uncased' for a white-box upper-bound "
                         "(matches TrustRAG's k-means encoder).")



args = parser.parse_args()

print(args)

os.makedirs(args.result_path, exist_ok=True)

custom_attack_path = args.custom_attack_path


with open(args.custom_attack_path, 'r') as file:
    attack_result = json.load(file)

with open(args.benign_score_path, 'r') as file:
    benign_scores = json.load(file)


with open(args.synth_docs_path, 'r') as file:
    synth_docs_container = json.load(file)


num_cand = args.num_cand
num_iter = args.num_iter
adv_per_query = args.adv_per_query
# total created adv text per query is adv_per_query * break_num.

device = 'cuda'
print("loading embedding model.")
embedding_model = SentenceTransformer(args.surrogate_model_name).to(device)
print(f"[surrogate] loaded {args.surrogate_model_name}")
embedding_tokenizer = embedding_model.tokenizer
print("loading embedding model finished.")
embedding_model.eval()



keys_list = list(attack_result.keys())

keys_list = keys_list[:args.data_num]
keys_list = sorted(keys_list)

print("Before keys_list len : ", len(keys_list))
if args.manual_world_size != -1:
    keys_list = [keys_list[idx] for idx in range(len(keys_list)) if idx % args.manual_world_size == args.manual_rank]

print("After keys_list len : ", len(keys_list))


if args.debug_mode:
    keys_list = keys_list[:2]

_FINAL_PATH = os.path.join(args.result_path, args.file_name)
_CKPT_PATH = _FINAL_PATH + '.ckpt'
_CHECKPOINT_EVERY = 5  # save checkpoint every N successfully-processed qids

if os.path.exists(_CKPT_PATH):
    try:
        with open(_CKPT_PATH, 'r') as _ckpt_f:
            _ckpt = json.load(_ckpt_f)
        if args.manual_world_size == -1:
            _completed_qids = {q for q, v in _ckpt.items()
                               if isinstance(v, dict) and v.get('_mix_completed', False)}
        else:
            _completed_qids = {q for q, v in _ckpt.items()
                               if isinstance(v, dict) and 'adv_texts_concat' in v}
        if _completed_qids:
            _n_before = len(keys_list)
            for _q in _completed_qids:
                if _q in attack_result:
                    attack_result[_q] = _ckpt[_q]
            keys_list = [q for q in keys_list if q not in _completed_qids]
            print(f"[resume] checkpoint at {_CKPT_PATH}: skipping "
                  f"{_n_before - len(keys_list)} already-completed qids; "
                  f"{len(keys_list)} remaining.")
        else:
            print(f"[resume] checkpoint at {_CKPT_PATH} found but no "
                  f"completed entries detected; starting fresh.")
    except Exception as _ckpt_err:
        print(f"[resume] could not parse checkpoint at {_CKPT_PATH}: "
              f"{type(_ckpt_err).__name__}: {_ckpt_err}; starting fresh.")


if args.coherence_filter:
    print("Coherence filter enabled.")
    ppl_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)
    ppl_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    ppl_model.eval()


## check whether all test keys are included in synth_docs

missing = [k for k in keys_list if k not in synth_docs_container]
if missing:
    sys.exit(
        f"{len(missing)} of {len(keys_list)} target queries are missing from "
        f"--synth_docs_path ({args.synth_docs_path}). First few: {missing[:5]}. "
        "The Stage-1 synthetic-document file must cover every target query."
    )



if torch.cuda.is_available():
    torch.cuda.synchronize()
script_start_time = time.perf_counter()
per_query_elapsed = []

keys_cnt = 0
for one_key in keys_list:
    print("keys_cnt : ", keys_cnt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    query_start_time = time.perf_counter()


    if one_key not in attack_result:
        print(f"[skip] {one_key} not in attack_result")
        continue
    if one_key not in benign_scores:
        print(f"[skip] {one_key} not in benign_scores")
        continue


    query = attack_result[one_key]['question']
    target_adv_texts = attack_result[one_key]['adv_texts'][:adv_per_query]


    benign_topk_docs_list = []

    synth_docs = synth_docs_container[one_key]['synth_texts']

    benign_topk_docs_list =  synth_docs


    if args.break_num is not None:
        benign_topk_docs_list, target_adv_texts = split_docs(benign_topk_docs_list, target_adv_texts, args.break_num)


    num_adv = len(target_adv_texts)
    print("num adv : ", num_adv)
    adv_texts=[]

    total_benign_texts = benign_topk_docs_list
    total_adv_texts = target_adv_texts

    keys_cnt += 1

    # attack
    for j in range(num_adv):
        print("adv idx : ", j)
        
        embeddings = get_embeddings(embedding_model)
        embedding_gradient = GradientStorage(embeddings) 

        target_query = attack_result[one_key]['question']


        for it_ in range(num_iter):
            print(f"(it_/num_iter) : {it_}/{num_iter}" )
            embedding_model.zero_grad()


            benign_inputs = embedding_tokenizer(total_benign_texts, return_tensors="pt", truncation=True, padding=True)
            benign_inputs = {k: v.cuda() for k, v in benign_inputs.items()}  

            benign_inputs.pop('token_type_ids', None)

            total_benign_embeddings = embedding_model(benign_inputs)['sentence_embedding']
            


            mean_embedding = torch.mean(total_benign_embeddings, dim=0, keepdim=True)
            distances = torch.norm(total_benign_embeddings - mean_embedding, dim=1)
            mean_dist = torch.mean(distances) # mean_dist_from the centroid


            total_dist = mean_dist

            print("total dist : ", total_dist.item())


            total_dist.backward()

            # Gradient w.r.t. the word-embedding output for document j,
            # captured by GradientStorage's backward hook.
            grad = embedding_gradient.get()[j]



            current_seq_len = benign_inputs["input_ids"][j].size(0)
            prevention_mask = torch.zeros(current_seq_len, dtype=torch.bool).cpu()

            token_to_flip, specials_id = pick_safe_token_to_flip(benign_inputs, j, embedding_tokenizer, prevention_mask = prevention_mask)
            print("chosen idx : ", token_to_flip)

            if args.coherence_filter:

                candidates = hotflip_attack(grad[token_to_flip],
                                            embeddings.weight,
                                            increase_loss=True,
                                            num_candidates=num_cand,
                                            filter=None,
                                            banned_ids=specials_id)


                candidates = candidate_filter(candidates,
                                    num_candidates=args.num_coherence_cand,
                                    token_to_flip=token_to_flip,
                                    adv_passage_ids=benign_inputs['input_ids'][j],
                                    ppl_model=ppl_model,
                                    cluster_embedding_tokenizer=embedding_tokenizer,
                                    ppl_tokenizer=ppl_tokenizer)

            else:

                candidates = hotflip_attack(grad[token_to_flip],
                                            embeddings.weight,
                                            increase_loss=True,
                                            num_candidates=num_cand,
                                            filter=None,
                                            banned_ids=specials_id)


            current_score = total_dist.sum().cpu().item()

            candidate_scores = torch.zeros(num_cand, device=device)
            before_id = int(benign_inputs['input_ids'][j, token_to_flip])

            for i, candidate in enumerate(candidates):
                temp_benign_inputs = embedding_tokenizer(total_benign_texts, return_tensors="pt", truncation=True, padding=True)
                temp_benign_inputs['input_ids'][j, token_to_flip] = candidate
                temp_benign_inputs = {k: v.cuda() for k, v in temp_benign_inputs.items()}
                temp_benign_inputs.pop('token_type_ids', None)
                with torch.no_grad():
                    temp_total_benign_embeddings = embedding_model(temp_benign_inputs)['sentence_embedding']

                    temp_mean_embedding = torch.mean(temp_total_benign_embeddings, dim=0, keepdim=True)
                    temp_distances = torch.norm(temp_total_benign_embeddings - temp_mean_embedding, dim=1)
                    temp_mean_dist = torch.mean(temp_distances)
                    temp_total_dist = temp_mean_dist

                    candidate_scores[i] += temp_total_dist


            if (candidate_scores > current_score).any().item():


                if keys_cnt % 10 == 0:
                    print("="*25, "Before updating", "="*25)
                    print(total_benign_texts[j])
                    print("="*50)


                best_candidate_idx = candidate_scores.argmax()
                temp_benign_inputs['input_ids'][j, token_to_flip] = candidates[best_candidate_idx]

                total_benign_texts[j] = embedding_tokenizer.decode(temp_benign_inputs["input_ids"][j].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)


                tok = embedding_tokenizer.convert_ids_to_tokens
                after_id = int(candidates[best_candidate_idx].item())
                after_tok  = tok([after_id])[0]

                before_tok = tok([before_id])[0]

                before_dec = embedding_tokenizer.decode([before_id])
                after_dec  = embedding_tokenizer.decode([after_id])

                print(f"[j={j} pos={token_to_flip}] {before_id}:{before_tok!r} ({before_dec!r})  ->  "
                    f"{after_id}:{after_tok!r} ({after_dec!r})")



                if keys_cnt % 10 == 0:
                    print("="*25, "After updating", "="*25)
                    print(total_benign_texts[j])
                    print("="*50)

        total_benign_texts[j] = total_benign_texts[j] + total_adv_texts[j] 

        adv_text = total_benign_texts[j]
        adv_texts.append(adv_text)

    if args.manual_world_size == -1:
        # Single-shard run: no merge step downstream, so write 'adv_texts' directly
        # (overwriting the loaded LLM drafts) so evaluate.py can read it as-is.
        attack_result[one_key]['adv_texts'] = adv_texts
        # Sentinel field for resume support — see the checkpoint block
        # earlier. The multi-shard path uses 'adv_texts_concat' presence
        # as the completion marker, so it doesn't need this field.
        attack_result[one_key]['_mix_completed'] = True
    else:
        # Multi-shard run: keep 'adv_texts_concat' so merge_shards.py
        # can pop the LLM drafts and rename 'adv_texts_concat' -> 'adv_texts'.
        attack_result[one_key]['adv_texts_concat'] = adv_texts


    # HotFlip wall-clock.
    if keys_cnt % _CHECKPOINT_EVERY == 0:
        _tmp_path = _CKPT_PATH + '.tmp'
        with open(_tmp_path, 'w') as _ckpt_f:
            json.dump(attack_result, _ckpt_f, indent=4)
        os.replace(_tmp_path, _CKPT_PATH)
        print(f"[ckpt] saved progress: keys_cnt={keys_cnt} -> {_CKPT_PATH}")

    # concatenate

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    query_elapsed = time.perf_counter() - query_start_time
    per_query_elapsed.append(query_elapsed)
    running_avg = sum(per_query_elapsed) / len(per_query_elapsed)
    remaining = max(0, len(keys_list) - len(per_query_elapsed))
    eta_seconds = running_avg * remaining
    print(f"[time] query={one_key} elapsed={query_elapsed:.2f}s | "
          f"avg/query={running_avg:.2f}s | done={len(per_query_elapsed)}/{len(keys_list)} | "
          f"ETA={eta_seconds/60:.1f}min")


if torch.cuda.is_available():
    torch.cuda.synchronize()
script_total_elapsed = time.perf_counter() - script_start_time
if per_query_elapsed:
    avg_q = sum(per_query_elapsed) / len(per_query_elapsed)
    min_q = min(per_query_elapsed)
    max_q = max(per_query_elapsed)
    print(f"[time-summary] total wall-clock = {script_total_elapsed:.1f}s ({script_total_elapsed/60:.2f} min)")
    print(f"[time-summary] per-query: n={len(per_query_elapsed)} avg={avg_q:.2f}s "
          f"min={min_q:.2f}s max={max_q:.2f}s")
else:
    print(f"[time-summary] total wall-clock = {script_total_elapsed:.1f}s (no queries processed)")


with open(args.result_path + "/" + args.file_name, 'w') as f:
    json.dump(attack_result, f, indent=4)

# Clean up the in-progress checkpoint now that the final output is on
# disk. Wrapped in try/except so any cleanup failure doesn't mark the
# run as failed (the canonical output is what matters).
try:
    if os.path.exists(_CKPT_PATH):
        os.remove(_CKPT_PATH)
        print(f"[ckpt] removed completed checkpoint: {_CKPT_PATH}")
except Exception as _rm_err:
    print(f"[ckpt] could not remove checkpoint: {_rm_err}")

print("the attack result is saved at ", args.result_path)
