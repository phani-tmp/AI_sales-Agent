# # Load synthetic dataset
# import json
# with open("car_insurance_synthetic_dataset_initiated_v2.json", "r") as f:
#     dataset = json.load(f)

# # Let's format the dataset into a structure suitable for training
# formatted_data = []

# # Loop through the dataset
# for entry in dataset:
#     conversation_flow = entry["conversation_flow"]
    
#     # Extract user inputs and agent responses
#     for turn in conversation_flow:
#         user_input = turn["input"]
#         agent_response = turn["output"]
        
#         formatted_data.append({
#             "input": user_input,
#             "output": agent_response
#         })

# # Check the first few samples
# # print(formatted_data[:3])

# from datasets import Dataset

# # Convert to a Hugging Face Dataset
# train_dataset = Dataset.from_dict({
#     "input": [entry["input"] for entry in formatted_data],
#     "output": [entry["output"] for entry in formatted_data]
# })

# # Check the dataset structure
# # print(train_dataset)
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import get_peft_model, LoraConfig, TaskType
# from torch.utils.data import DataLoader
# from torch.optim import AdamW
# from tqdm import tqdm
# import torch

# # Load the base model and tokenizer
# model_name = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Initialize LoRA config
# lora_config = LoraConfig(
#     r=8,  # Low-rank matrix size
#     lora_alpha=16,  # Scaling factor for LoRA layers
#     lora_dropout=0.1,  # Dropout rate for LoRA layers
#     task_type=TaskType.CAUSAL_LM,
# )

# # Apply LoRA to the model
# peft_model = get_peft_model(model, lora_config)

# # Ensure model is on the correct device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# peft_model.to(device)

# # Define DataLoader (Ensure `train_dataset` is properly preprocessed)
# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# # Set up optimizer
# optimizer = AdamW(peft_model.parameters(), lr=5e-5)

# # Fine-tuning loop
# epochs = 3
# for epoch in range(epochs):
#     peft_model.train()
#     loop = tqdm(train_dataloader, leave=True)

#     for batch in loop:
#         optimizer.zero_grad()

#         # Ensure batch["input_ids"] is a tensor
#         if isinstance(batch["input_ids"], list):
#             inputs = torch.stack([torch.tensor(x, dtype=torch.long) for x in batch["input_ids"]]).to(device)
#         else:
#             inputs = batch["input_ids"].to(device)

#         if isinstance(batch["labels"], list):
#             labels = torch.stack([torch.tensor(x, dtype=torch.long) for x in batch["labels"]]).to(device)
#         else:
#             labels = batch["labels"].to(device)

#         # Forward pass
#         outputs = peft_model(input_ids=inputs, labels=labels)
#         loss = outputs.loss

#         # Backward pass
#         loss.backward()
#         optimizer.step()

#         loop.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

# # Save the fine-tuned model
# peft_model.save_pretrained("fine_tuned_car_insurance_model")
# tokenizer.save_pretrained("fine_tuned_car_insurance_model") 
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# ✅ **Load synthetic dataset**
with open("car_insurance_synthetic_dataset_initiated_v2.json", "r") as f:
    dataset = json.load(f)

# ✅ **Format dataset for training**
formatted_data = []
for entry in dataset:
    for turn in entry["conversation_flow"]:
        formatted_data.append({
            "input": turn["input"],
            "output": turn["output"]
        })

# ✅ **Load tokenizer**
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding is handled

# ✅ **Custom Dataset Class**
class InsuranceDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = conversations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_text = f"User: {sample['input']} \nAgent: {sample['output']}"

        # ✅ **Tokenize input and output**
        encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0),  # Causal LM expects labels = input_ids
        }

# ✅ **Convert dataset to PyTorch Dataset**
train_dataset = InsuranceDataset(formatted_data, tokenizer)

# ✅ **Define DataLoader**
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# ✅ **Load base model**
model = AutoModelForCausalLM.from_pretrained(model_name)

# ✅ **Initialize LoRA config**
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)

# ✅ **Apply LoRA to model**
peft_model = get_peft_model(model, lora_config)

# ✅ **Ensure model is on the correct device**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
peft_model.to(device)

# ✅ **Set up optimizer**
optimizer = AdamW(peft_model.parameters(), lr=5e-5)

# ✅ **Fine-tuning loop**
epochs = 3
for epoch in range(epochs):
    peft_model.train()
    loop = tqdm(train_dataloader, leave=True)

    for batch in loop:
        optimizer.zero_grad()

        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # ✅ **Forward pass**
        outputs = peft_model(input_ids=inputs, labels=labels)
        loss = outputs.loss

        # ✅ **Backward pass**
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

# ✅ **Save the fine-tuned model**
peft_model.save_pretrained("fine_tuned_car_insurance_model")
tokenizer.save_pretrained("fine_tuned_car_insurance_model")

print("Fine-tuning complete! Model saved.")
