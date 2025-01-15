
import functools
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from opencoconut import AutoCoconutForCausalLM, CoTDataset, split_sequences
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch

def extract_generated_answer(model_output: str, eos_token="<|im_end|>"):
    answer_prefix = "Answer: "
    start_index = model_output.find(answer_prefix)

    if start_index == -1:
        return None

    start_index += len(answer_prefix)
    end_index = model_output.find(eos_token, start_index)

    if end_index == -1:
        return None

    extracted_answer = model_output[start_index:end_index].strip()
    return extracted_answer

def _pp(s, tokenizer):
    if s is None:
        return s
    s = s.replace(tokenizer.eos_token, '')
    s = s.replace(tokenizer.bos_token, '')
    s = s.replace(tokenizer.pad_token, '')
    return s


@torch.no_grad()
def evaluate(
    dataloader,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    max_new_tokens: int,
    verbose = 1,
    add_bot = False
):
    pp = functools.partial(_pp, tokenizer=tokenizer)
    total_instances = 0
    total_correct = 0
    for batch in tqdm(dataloader):
        if add_bot:
            batch["input_ids"], batch["attention_mask"] = model.append_bot_token(batch["input_ids"], batch["attention_mask"])

        (
            thought_ids,
            language_ids,
            thought_mask,
            _,
            _,
            _,
        ) = split_sequences(**batch, coconut_config=model.coconut_config)
        batch_size = thought_ids.shape[0]
        total_instances += batch_size

        # Generate
        beam_output = model.generate(
            input_ids=thought_ids.to(model.device),
            attention_mask=thought_mask.to(model.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            # use_cache=False,
        )
        # Evaluate
        for thought_ids_batch, output_batch in zip(thought_ids, beam_output):
            decoded_language_ids = tokenizer.decode(language_ids[0])
            decoded_pred_text = tokenizer.decode(output_batch)
            answer = extract_generated_answer(
                decoded_language_ids, eos_token=tokenizer.eos_token
            )
            pred_answer = extract_generated_answer(
                decoded_pred_text, eos_token=tokenizer.eos_token
            )
            if answer == pred_answer:
                total_correct += 1
            if verbose>1:
                print(
                    f"Input: `{pp(tokenizer.decode(thought_ids_batch, skip_special_tokens=True))}`\n"
                    f"decoded_language_ids: `{pp(decoded_language_ids)}`\n"
                    f"decoded_pred_text: `{pp(decoded_pred_text)}`\n"
                    f"Target: `{pp(answer)}`\n"
                    f"Predicted: `{pp(pred_answer)}`\n"
                )
    if verbose>0:
        print(
            f"Input: `{pp(tokenizer.decode(thought_ids_batch, skip_special_tokens=True))}`\n"
            f"decoded_language_ids: `{pp(decoded_language_ids)}`\n"
            f"decoded_pred_text: `{pp(decoded_pred_text)}`\n"
            f"Target: `{pp(answer)}`\n"
            f"Predicted: `{pp(pred_answer)}`\n"
        )
    accuracy = total_correct / total_instances
    return accuracy
