import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use YOUR modules
from energypeft.core.carbon_scheduler import wait_for_green_grid
from energypeft.core.energy_monitor import EnergyMonitor
from energypeft.core.efficient_training import EnergyAwareTrainingController, attach_indices_collate


# -----------------------
# 1) Dataset wrapper
# -----------------------
class DictDataset(Dataset):
    """
    Wraps a list of dict samples and injects _index so efficient_training can
    track scores per sample and update them over time.
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = dict(self.samples[idx])
        item["_index"] = idx  # required for controller feedback
        return item


# -----------------------
# 2) Per-sample loss for CausalLM (vector, not scalar)
# -----------------------
def per_sample_causal_lm_loss(logits, labels, attention_mask):
    """
    logits: [B, T, V]
    labels: [B, T]
    attention_mask: [B, T]  (1 real, 0 pad)

    Returns: loss_per_sample [B]
    """
    # Shift for causal LM
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # Token-level cross entropy (no reduction)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())  # [B, T-1]

    # Mask out padding tokens
    token_loss = token_loss * shift_mask

    # Per-sample average over real tokens
    denom = shift_mask.sum(dim=1).clamp(min=1)
    loss_per_sample = token_loss.sum(dim=1) / denom
    return loss_per_sample


def main():
    # -----------------------
    # 0) Carbon scheduler
    # -----------------------
    wait_for_green_grid(max_intensity=250, region="CA-QC")

    # -----------------------
    # 1) Model & tokenizer
    # -----------------------
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    # -----------------------
    # 2) Dummy dataset (your pattern)
    # -----------------------
    samples = []
    for i in range(100):
        # Make random token ids, simulate padding
        input_ids = torch.randint(0, 1000, (64,), dtype=torch.long)
        labels = torch.randint(0, 1000, (64,), dtype=torch.long)

        attention_mask = torch.ones(64, dtype=torch.long)
        attention_mask[-10:] = 0  # pretend last 10 are pad

        # set pad tokens in input_ids/labels to pad_token_id for realism (optional)
        pad_id = tokenizer.pad_token_id
        input_ids[-10:] = pad_id
        labels[-10:] = -100  # ignore index convention for LM labels (optional)

        samples.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        )

    dataset = DictDataset(samples)

    # -----------------------
    # 3) Energy monitor (energy budget is now a constraint)
    # -----------------------
    monitor = EnergyMonitor(energy_budget_wh=50.0, cpu_backend="auto")
    monitor.start_monitoring()

    # -----------------------
    # 4) Controller (sampler + adaptive batch + early stopper)
    # -----------------------
    controller = EnergyAwareTrainingController(
        dataset=dataset,
        energy_monitor=monitor,
        base_batch_size=16,
        min_batch_size=4,
        max_steps=500,
    )

    # IMPORTANT: use batch_sampler for dynamic batch size
    # Note: Do NOT pass batch_size, shuffle, sampler, or drop_last when using batch_sampler
    loader = DataLoader(
        dataset,
        batch_sampler=controller.batch_sampler,
        collate_fn=attach_indices_collate,
        num_workers=0,
    )

    # -----------------------
    # 5) Optimizer
    # -----------------------
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    pad_token_id = tokenizer.pad_token_id

    # Optional: validation stub (replace with real val loop)
    best_val = None

    # -----------------------
    # 6) Training loop
    # -----------------------
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_indices = batch["_indices"]  # list of dataset indices

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, T, V]

        # 6.1 compute per-sample loss vector [B]
        # If you used -100 in labels for padding, create a mask consistent with it:
        # Here we use attention_mask, and treat label=-100 as pad too.
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = pad_token_id
        per_sample_loss = per_sample_causal_lm_loss(logits, safe_labels, attention_mask)

        # 6.2 token lengths [B] = cost proxy
        lengths = attention_mask.sum(dim=1)

        # 6.3 backward using mean loss
        loss = per_sample_loss.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # 6.4 feedback to controller (updates sampling scores + logs batch size/energy)
        controller.on_train_step_end(
            batch_indices=batch_indices,
            per_sample_losses=per_sample_loss,
            lengths=lengths,
            global_step=step,
        )

        # 6.5 early stopping (energy + plateau)
        # For a real run: compute a real validation metric every N steps.
        # For demo: use negative training loss as a fake "higher is better" metric.
        val_metric = float((-loss).detach().cpu())

        if controller.early_stopper.should_stop(
            step=step,
            remaining_energy_wh=monitor.get_remaining_energy(),
            val_metric=val_metric,
        ):
            print(f"Early stop at step={step} | remaining={monitor.get_remaining_energy():.4f} Wh")
            break

        if step % 20 == 0:
            print(
                f"step={step} | loss={float(loss):.4f} | remaining={monitor.get_remaining_energy():.3f} Wh"
            )

    # -----------------------
    # 7) Stop monitoring + save log
    # -----------------------
    metrics = monitor.stop_monitoring()
    monitor.save_energy_log("energy_log.json")
    print(f"Total energy used (Wh): {metrics.total_energy_wh:.4f}")
    print("Saved: energy_log.json")


if __name__ == "__main__":
    main()
