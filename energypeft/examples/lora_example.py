import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from energypeft import EnergyPEFT

# ---------------------------------------------------------
# HELPER: Data Pre-processing
# ---------------------------------------------------------
def get_processed_dataset(tokenizer, num_samples=500):
    """
    Loads, tokenizes, and indexes data for Green PEFT testing.
    """
    print(f"📥 Loading first {num_samples} samples from yahma/alpaca-cleaned...")
    # 1. Load raw data
    raw_dataset = load_dataset("yahma/alpaca-cleaned", split=f"train[:{num_samples}]")

    # 2. Define how to format and tokenize the text
    def tokenize_function(examples):
        # Combine instruction + input + output into one prompt
        # We handle cases where 'input' might be empty
        prompts = []
        for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output']):
            if inp:
                text = f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
            else:
                text = f"Instruction: {inst}\nOutput: {out}"
            prompts.append(text)
        
        # Tokenize!
        # max_length=512 keeps sequences reasonable for testing
        return tokenizer(prompts, truncation=True, max_length=512, padding="max_length")

    print("⚙️ Tokenizing data (Converting text to numbers)...")
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

    # 3. CRITICAL: Add Index Column
    # The GreenTrainer/SmartSampler needs this 'index' to track which sample is which
    print("🔢 Adding indices for energy tracking...")
    tokenized_dataset = tokenized_dataset.map(lambda x, idx: {"index": idx}, with_indices=True)
    
    # 4. Cleanup: Keep only the columns the model needs
    # We keep 'input_ids', 'attention_mask', 'labels' (created by trainer usually, but we reuse input_ids here)
    # AND 'index' for our custom logic.
    # Note: For Causal LM, labels are usually the same as input_ids.
    def add_labels(examples):
        examples["labels"] = examples["input_ids"]
        return examples
        
    tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
    
    # Set format for PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "index"])
    
    print(f"✅ Ready! Dataset has {len(tokenized_dataset)} samples.")
    return tokenized_dataset

# ---------------------------------------------------------
# MAIN TRAINING SCRIPT
# ---------------------------------------------------------
def main():
    # 1. Initialize Energy Framework
    energy_peft = EnergyPEFT(energy_budget_wh=30.0)
    
    # 2. Load Model & Tokenizer
    #model_name = "meta-llama/Llama-3.1-8B-Instruct" 
    # Note: If you don't have access to Llama 3.1, use "openlm-research/open_llama_3b" for a smaller free test
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Critical for LLaMA
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 3. Prepare Data
    train_dataset = get_processed_dataset(tokenizer, num_samples=500)
    
    # 4. Apply LoRA
    print("🛠️ Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"], # Reduced targets for faster testing
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # 5. Wrap with GreenTrainer
    # This automatically injects your GreenTrainer and SmartSampler logic
    trainer = energy_peft.wrap_trainer(
        trainer_type="huggingface",
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        # Standard HF Args
        args=TrainingArguments(
            output_dir="./green_results",
            per_device_train_batch_size=2, # Small batch for testing
            num_train_epochs=1,
            logging_steps=10,
            learning_rate=2e-4,
            remove_unused_columns=False # IMPORTANT: Keeps our 'index' column!
        )
    )
    
    # 6. Train!
    print("⚡ Starting Green PEFT Training...")
    trainer.train()
    print("🎉 Training Finished!")

if __name__ == "__main__":
    main()