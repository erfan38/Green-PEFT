# examples/api_demo.py
"""
EnergyPEFT API Usage Examples
Shows current working API and future planned API
"""

def current_api_example():
    """Example using current EnergyPEFT API"""
    import energypeft
    
    # Current working API
    energy_peft = energypeft.EnergyPEFT(energy_budget_wh=100.0)
    
    # This works with your current code
    trainer = energy_peft.wrap_trainer(
        trainer_type="huggingface",
        # model=your_model,
        # tokenizer=your_tokenizer,  
        # train_dataset=your_dataset
    )
    
    print("✅ Current API demo - EnergyPEFT framework created!")
    return trainer

def future_api_example():
    """Example of planned future API"""
    # Future API (when GreenTrainer is ready)  
    # trainer = energypeft.GreenTrainer(
    #     model=model, 
    #     args=training_args, 
    #     train_dataset=dataset, 
    #     energy_budget_wh=100.0
    # )
    
    print("🔮 Future API demo - GreenTrainer (coming soon!)")

if __name__ == "__main__":
    print("🌱 EnergyPEFT API Examples")
    print("=" * 30)
    
    current_api_example()
    future_api_example()
