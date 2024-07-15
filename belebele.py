from datasets import load_dataset
from gpt2 import GPT, GPTConfig
from transformers import GPT2LMHeadModel
from torch.nn import functional as F
import torch
import tiktoken

# hf download does not work due to bad zip file
ds = load_dataset("facebook/belebele", split="eng_Latn")

# loading with manual download
# ds = load_dataset("json", data_files="pol_Latn.jsonl")["train"]

enc = tiktoken.get_encoding("gpt2")
model = GPT(GPTConfig(vocab_size=50304))
# model = model = GPT2LMHeadModel.from_pretrained("gpt2")

def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["question"]
    label = int(example["correct_answer_num"]) - 1
    example["endings"] = [example["mc_answer1"], example["mc_answer2"], example["mc_answer3"], example["mc_answer4"]]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # note: prepending " " because GPT-2 tokenizer -- may be treated as a sentence start and get a special token
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples():
    for example in ds:
        yield example

@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision('high') # use tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model) # optionally torch compile the model

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples():
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).logits # size: sentences x tokens x vocab
        shift_logits = (logits[..., :-1, :]).contiguous() # size: sentences x (tokens-1) x vocab
        shift_tokens = (tokens[..., 1:]).contiguous()     # size: sentences x (tokens-1)
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # size: (sentences*tokens) x vocab
        flat_shift_tokens = shift_tokens.view(-1)                        # size: (sentences*tokens)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1) # size: sentences x tokens
        shift_mask = (mask[..., 1:]).contiguous() # size: sentences x (tokens-1)
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f} -- pred: {pred}, label: {example['correct_answer_num']}")

        if num_total < 10:
            print("---")
            print(f"Context:\n {example['question']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

    print(f"Final acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

evaluate("gpt2", "cuda")
