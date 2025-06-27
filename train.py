from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

# 1. Prepare tiny dataset of prompts and completions
train_data = [
    {"prompt": "Hello, how are", "completion": " you today?"},
    {"prompt": "The capital of France is", "completion": " Paris."},
    {"prompt": "PyTorch is a popular", "completion": " deep learning library."},
]

# 2. Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 3. Tokenize dataset
class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.examples = []
        for item in data:
            enc = tokenizer(item["prompt"] + item["completion"], truncation=True, max_length=32)
            self.examples.append(torch.tensor(enc["input_ids"]))
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

dataset = TinyDataset(train_data)

# 4. Define training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=1,
    logging_dir="./logs",
    save_steps=10,
    save_total_limit=1,
)

# 5. Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 6. Run training
trainer.train()

# 7. Save model + tokenizer
model.save_pretrained("./mini-gpt2-finetuned")
tokenizer.save_pretrained("./mini-gpt2-finetuned")

print("Training complete! Model saved to ./mini-gpt2-finetuned")
