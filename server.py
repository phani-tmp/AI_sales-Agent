from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
import callAgent  # Import AI assistant logic
import os 

app = Flask(__name__)

@app.route("/voice", methods=['POST'])
def voice():
    """Handles incoming Twilio calls and processes user input dynamically."""
    response = VoiceResponse()
    
    # Get user's speech-to-text (from Twilio)
    user_input = request.form.get('SpeechResult', '')

    print(f"User said: {user_input}")  # Debugging

    if user_input:
        # Get AI-generated response from Maya
        ai_response = callAgent.handle_user_query(user_input)
        
        # Respond with AI-generated speech
        response.say(ai_response, voice="alice")
    
    # Continue the conversation
    response.gather(input='speech', action='/voice', method='POST', timeout=5)

    return Response(str(response), mimetype="text/xml")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
