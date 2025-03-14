from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load fine-tuned model
model_path = "fine_tuned_car_insurance_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate responses
def chat_with_model(user_input):
    input_text = f"User: {user_input} \nAgent:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    output_ids = model.generate(
        **inputs, 
        max_length=150, 
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,  # Prevents repeated n-grams
        do_sample=True,          # Enables sampling for variability
        temperature=0.7,         # Adjusts randomness (lower = more deterministic)
        top_k=50,                # Limits sampling to top-k words
        top_p=0.9                # Nucleus sampling for diversity
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return response


# Example interaction
user_message = "Full coverage sounds too expensive."
response = chat_with_model(user_message)
print("Chatbot Response:", response)
