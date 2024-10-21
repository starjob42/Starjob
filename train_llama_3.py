import argparse
import torch
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a FastLanguageModel with specified parameters.")
    
    # Model and data parameters
    parser.add_argument('--max_seq_length', type=int, default=50000, help='Maximum sequence length')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'], help='Data type (bfloat16 or float16)')
    parser.add_argument('--load_in_4bit', action='store_true',default=True, help='Use 4-bit quantization to reduce memory usage')

    # LoRA hyperparameters
    parser.add_argument('--lora_r', type=int, default=64, help='Rank of the LoRA decomposition')
    parser.add_argument('--lora_alpha', type=int, default=64, help='Scaling factor for LoRA updates')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='Dropout rate for LoRA layers')
    parser.add_argument('--bias', type=str, default='none', choices=['none', 'all', 'lora_only'], help='Bias type')

    # Additional configurations
    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth', help='Use gradient checkpointing')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--use_rslora', action='store_true',default=False, help='Use RSLoRA')
    parser.add_argument('--loftq_config', type=str, default=None, help='LoFT-Q configuration')

    # Training hyperparameters
    parser.add_argument('--per_device_train_batch_size', type=int, default=4, help='Batch size per device during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=5, help='Number of warmup steps')
    parser.add_argument('--num_train_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--logging_steps', type=int, default=1, help='Logging steps')
    parser.add_argument('--optim', type=str, default='adamw_8bit', help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Learning rate scheduler type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_total_limit', type=int, default=50, help='Total save limit for model checkpoints')
    parser.add_argument('--save_step', type=int, default=200, help='Steps interval to save model checkpoints')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2, help='Batch size per device during evaluation')
    parser.add_argument('--train_lm_head', action='store_true',default=False, help='Weather to train the language model head or not')
    parser.add_argument('--train_embed_tokens', action='store_true',default=False, help='Weather to train the embed_tokens or not')

    # Output directory
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name')

    args = parser.parse_args()

    # =========================
    # Generate Output Directory Name
    # =========================

    # Create an output directory name based on hyperparameters
    if args.output_dir is None:
        dir_out = f"output_alpha{args.lora_alpha}_r{args.lora_r}_train_lm_head{args.train_lm_head}_train_embed_tok_{args.train_embed_tokens}_seq{args.max_seq_length}_b{args.per_device_train_batch_size}_ep{args.num_train_epochs}"
    else:
        dir_out = args.output_dir

    # =========================
    # Initialize WandB
    # =========================

    # Initialize Weights & Biases for experiment tracking
    wandb.init(
        project="llama3-jssp-clean",  # Change the project name if needed
        name=dir_out,
    )

    # =========================
    # Load Model and Tokenizer
    # =========================

    # Set dtype
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    # Load the pre-trained model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )
    
    target_modules =[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", 
        ]
    if args.train_lm_head:
        target_modules.append('lm_head')
    if args.train_embed_tokens:
        target_modules.append('embed_tokens')

    # Configure the model with PEFT (Parameter-Efficient Fine-Tuning)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        loftq_config=args.loftq_config,
    )

    # Define the Alpaca-style prompt template
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    # =========================
    # Load and Prepare Dataset
    # =========================

    #put the data in the data folder
    dataset = load_dataset('./data/', split="train")
    split_dataset = dataset.train_test_split(test_size=0.02, seed=args.seed)
    train_dataset = split_dataset['train'].map(formatting_prompts_func, batched=True)
    eval_dataset = split_dataset['test'].map(formatting_prompts_func, batched=True)

    # =========================
    # Initialize the Trainer
    # =========================

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=20,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            # fp16=True,
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=dir_out,
            report_to="wandb",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=args.save_total_limit,
            save_steps=args.save_step,
            eval_strategy="steps",
            eval_steps=args.save_step,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
        ),
    )

    # =========================
    # Monitor GPU Memory Usage
    # =========================

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # =========================
    # Start Training
    # =========================

    trainer_stats = trainer.train()

if __name__ == "__main__":
    main()
