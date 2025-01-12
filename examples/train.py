import os
import torch
import logging
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)
from opencoconut import (
    AutoCoconutForCausalLM,
    CoconutConfig,
    CoTDataset,
)
from pathlib import Path
import gc

# Configure logging
from loguru import logger

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def train():
    config_small = {
        'model_name': "Qwen/Qwen2.5-0.5B",
        'batch_size': 8,
        'learning_rate': 1e-4,
        'steps': 1000,
        'output_dir': "./output/small",
    }
    config_medium = {
        'model_name': "Qwen/Qwen2.5-2.5B",
        'batch_size': 1,
        'learning_rate': 5e-5,
        'steps': 30000,
        'output_dir': "./output/small",
    }
    config = config_small

    # Initialize model and tokenizer
    logger.info("Initializing model and tokenizer")
    output_dir = Path(config['output_dir'])

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|endoftext|>"

    # config and model
    config = CoconutConfig.from_tokenizer(
        tokenizer,
        stages=4,
        continuous_thoughts=2,
    )
    model = AutoCoconutForCausalLM.from_pretrained(
        config['model_name'], config, torch_dtype=torch.bfloat16, device_map=get_device()
    )
    model.resize_token_embeddings(len(tokenizer))
    if os.getenv("DEBUG", "0") == "1":
        model.tokenizer = tokenizer

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=1,
        learning_rate=config['learning_rate'],
        warmup_ratio=0.1,
        max_steps=config['steps'],
        logging_steps=10,
        save_steps=10000,
        bf16=True,
        bf16_full_eval=True,
        optim="adamw_torch", # save memory: adamw_bnb_8bit
    )

    # Initialize trainer
    for stage in range(config.stages):
        logger.info(f"starting stage {stage}")
        logger.info("preparing dataset")
        dataset = CoTDataset(
            "casperhansen/gsm8k_synthetic_cot",
            tokenizer,
            max_length=512, # 256 should be fine?
            coconut_config=config,
            current_stage=stage,
        )
        logger.info(f"dataset size: {len(dataset)}")
        model.current_stage = stage
        current_output_dir = output_dir/"stage{stage}"
        training_args.output_dir = current_output_dir

        if stage == 0:
            training_args.num_train_epochs = 6
        elif stage == config.stages-2:
            # Penultimate stage removes all the remaining language reasoning chain
            # This handles the long-tail distribution of reasoning chains longer than 3 steps
            dataset.include_reasoning_steps = False
            training_args.num_train_epochs = 3
        elif stage == config.stages-1:
            # For all datasets, after the standard schedule,
            # the model stays in the final training stage, until the 50th epoch.
            dataset.include_reasoning_steps = True
            training_args.num_train_epochs = 50

        logger.info("starting training")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

        # save tokenizer to all checkpoints after training
        for folder in os.listdir(current_output_dir):
            if folder.startswith("checkpoint-"):
                checkpoint_folder = os.path.join(current_output_dir, folder)
                if os.path.isdir(checkpoint_folder):
                    tokenizer.save_pretrained(checkpoint_folder)

        # save final checkpoint
        current_output_dir = f"{output_dir}/stage{stage}/final"
        model.save_pretrained(current_output_dir)
        tokenizer.save_pretrained(checkpoint_folder)
        logger.info(f"finished stage {stage}")

        clear_memory()




if __name__ == "__main__":
    train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    accuracy = evaluate(dataloader, tokenizer, model, args.max_new_tokens)
    print(f"Accuracy: {accuracy}")
