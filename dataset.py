import random
import json

# Define multiple variations for each part of the conversation
greetings = [
    "Hello! I'm Maya, a professional car insurance agent from AIphi. How are you today?",
    "Hi! I'm Maya from AIphi, here to assist you with your car insurance needs. How can I help?",
    "Good day! I’m Maya, your car insurance expert. Let’s talk about the best options for you!",
    "Hello, I’m Maya! I can guide you through the best insurance options for your car. How are you today?",
    "Hey there! I’m Maya from AIphi. Let’s find the right car insurance plan for you.",
    "Hi! This is Maya, I’m a car insurance agent at AIphi. How can I assist you today?",
    "Good morning, this is Maya with AIphi. I’d love to help you with car insurance today!",
    "Hello, I’m Maya from AIphi. How can I help you get the best insurance for your car?",
    "Hi, it’s Maya from AIphi! I’m here to help with your car insurance needs.",
    "Hello! Maya here, I’m your AI car insurance agent. What can I assist you with today?"
]

info_gathering = [
    "Got it! A 2018 Toyota Camry. Thanks for the info!",
    "I see you have a 2018 Toyota Camry. Let's move on!",
    "Thanks for sharing! A 2018 Toyota Camry, got it!",
    "Great, a 2018 Toyota Camry. What else can I assist you with?",
    "Got it, 2018 Toyota Camry. Let's continue!",
    "Thanks! A 2018 Toyota Camry. Next, we can look at insurance options."
]

insurance_options = [
    "Based on your clean record, I recommend our basic liability plan or a full coverage plan that includes theft protection and accident coverage. What would you prefer?",
    "I have two great options for you. A basic liability plan or comprehensive coverage, which includes accident and theft protection. Which would you like?",
    "We can go with a low-cost liability insurance plan or, if you prefer more coverage, our full plan that includes damages, theft, and accidents. Which one sounds better?",
    "For someone with your profile, I'd recommend either basic liability coverage or full coverage, which includes accident protection and theft coverage. What would you prefer?",
    "You could go for the basic liability or full coverage, which gives more protection in the event of an accident or theft. What works for you?",
    "If you're looking for something affordable, basic liability coverage might be the way to go. But I’d suggest full coverage for more protection. How do you feel about that?",
    "We offer two plans: basic liability or full coverage. Full coverage would protect you from accidents and theft. What sounds better?",
    "You have two great options: liability insurance or full coverage, which includes protection for accidents and theft. Which one fits your needs?",
    "We offer basic liability insurance or comprehensive coverage. The comprehensive coverage includes more protection. Which option suits you best?",
    "With your clean record, we recommend our basic liability insurance or our full coverage plan that includes accident and theft protection. What would you prefer?"
]

objections = [
    "I understand that you might have concerns about cost. We have flexible payment plans that fit any budget. Would you like to see some options?",
    "I hear you! We can explore cheaper plans with just the coverage you need. Would you like to discuss payment options?",
    "I get it! Full coverage may seem expensive, but we can customize a plan that fits your budget. Would you like to explore options?",
    "I understand your concern about price. Let me show you payment plans that can fit your needs. How about that?",
    "I see your concern. But the peace of mind that comes with full coverage is invaluable. How about I show you some affordable payment options?",
    "I know cost can be a factor, but we can provide options that match your budget. How about I suggest a few plans?",
    "We can definitely find a plan that fits your financial situation. Should I walk you through some more affordable options?",
    "It’s understandable to be cautious about cost. Let’s explore some cost-effective plans that offer good coverage.",
    "I understand the price concern. How about I show you the best value options for the protection you need?",
    "I see what you mean. We can work with you to find a plan that fits your budget. Would you like me to suggest some more affordable options?"
]

closing = [
    "Great! I’ll get your policy set up now. Let's proceed with the paperwork and payment details.",
    "Wonderful! I’ll move forward with your full coverage plan. Let’s finalize everything.",
    "Awesome choice! I’ll begin the process to finalize your insurance. Let’s set up your payment details.",
    "Fantastic! I’ll take care of the rest and send you the final paperwork.",
    "Perfect! I’ll set up your full coverage now. You’ll be fully protected. Let’s get the final details.",
    "Great! I’ll get everything ready for you. You’ll be covered in no time.",
    "All right, I’ll finalize the details now. You’re all set to go!",
    "Thanks for choosing our full coverage plan. I’ll get everything in order for you right now.",
    "Great! I’ll begin the process of securing your insurance. Let's finalize the paperwork.",
    "Awesome! I’ll take care of the details and send you everything you need to get started."
]

# Function to generate a synthetic conversation with agent initiating
def generate_conversation():
    greeting = random.choice(greetings)
    car_info = random.choice(info_gathering)  # Ensure fixed year here (2018 Toyota Camry)
    insurance_option = random.choice(insurance_options)
    objection = random.choice(objections)
    closing_message = random.choice(closing)
    
    conversation = {
        "input": "Hello, I'm looking for car insurance.",
        "output": greeting
    }
    
    conversation["conversation_flow"] = [
        {
            "input": "Hello, I'm looking for car insurance.",
            "output": greeting
        },
        {
            "input": "I drive a 2018 Toyota Camry.",
            "output": car_info
        },
        {
            "input": "I’ve never had an accident or traffic violation.",
            "output": insurance_option
        },
        {
            "input": "Full coverage sounds too expensive.",
            "output": objection
        },
        {
            "input": "I’m ready to proceed with the full coverage plan.",
            "output": closing_message
        }
    ]
    return conversation

# Generate a dataset with 100 conversations
dataset = [generate_conversation() for _ in range(100)]

# Save the dataset to a JSON file without any random years and without Unicode escape characters
with open('car_insurance_synthetic_dataset_fixed_year.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print("Synthetic dataset generated and saved with fixed year (2018) and no Unicode escapes!")
