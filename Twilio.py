import os
from twilio.rest import Client
from dotenv import load_dotenv  # Optional: For security

# Load environment variables from a .env file (if using)
load_dotenv()

# Twilio API Credentials
TWILIO_ACCOUNT_SID = "AC2780fbba0da5494f77fed33d3bd5ee9f"
TWILIO_AUTH_TOKEN = "d108b47ac554c71b4dd0fef391e1edee"  # Replace with your actual Auth Token
TWILIO_PHONE_NUMBER = "+17345304114"  # Replace with your Twilio Number

# Initialize Twilio Client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def make_call(to_number):
    """Make a voice call using Twilio"""
    call = client.calls.create(
        twiml='<Response><Say>Hello! This is a test call from AIphi.</Say></Response>',
        to=to_number,
        from_=TWILIO_PHONE_NUMBER
    )
    print(f"Call initiated. Call SID: {call.sid}")

# Replace with your actual phone number
make_call("+12403554770")  # Use your verified number (with country code)
