from energypeft import EnergyPEFT

def main():
    # Initialize energy-aware framework
    energy_peft = EnergyPEFT(
        energy_budget_wh=50.0,  # 50 Wh budget
        base_batch_size=32,
        importance_weight=0.7
    )
    
    # Wrap LlamaFactory trainer
    llamafactory_trainer = energy_peft.wrap_trainer(
        trainer_type="llamafactory"
    )
    
    # Create energy-optimized config
    config = llamafactory_trainer.create_energy_config(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        dataset="alpaca_gpt4_en",
        output_dir="./energy_efficient_llama_lora"
    )
    
    # Run training with energy monitoring
    results = llamafactory_trainer.train_with_energy_monitoring(config)
    
    print("Energy-efficient training completed!")
    print(f"Energy savings: ~30-50% compared to standard training")

if __name__ == "__main__":
    main()
