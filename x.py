# import threading
# from langchain.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.chat_history import InMemoryChatMessageHistory
# import chromadb
# from sentence_transformers import SentenceTransformer, util
# from twilio.rest import Client
# from flask import Flask, request
# import os
# import streamlit as st
# from transformers import pipeline

# # ✅ Twilio credentials
# TWILIO_ACCOUNT_SID = "AC2780fbba0da5494f77fed33d3bd5ee9f"
# TWILIO_AUTH_TOKEN = "49d878e96486dcdc30a98e9035019b81"  # Replace with your actual Auth Token
# TWILIO_PHONE_NUMBER = "+17345304114"  # Replace with your Twilio Number

# client_twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# # ✅ Initialize Groq LLM
# groq_api_key = "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"
# llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# # ✅ Connect to Chroma DB
# client = chromadb.PersistentClient(path="./chroma_db")
# collection = client.get_collection("geico_faqs")
 
# # ✅ Load embedding model
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # ✅ Define prompt template
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", """You are Maya, a professional car insurance agent. Your job is to help users 
#     understand car insurance options, guide them to the best plans, and explain benefits in simple terms."""),
#     ("human", "{input}"),
# ])

# # ✅ Initialize conversation memory
# message_history = InMemoryChatMessageHistory()

# # ✅ Function to fetch the best-matching insurance info from Chroma DB
# def retrieve_insurance_info(user_query, threshold=0.4):
#     query_vector = embed_model.encode(user_query)
    
#     # Query ChromaDB to get top 1 best match
#     results = collection.query(query_embeddings=[query_vector.tolist()], n_results=1)

#     if results['documents']:
#         best_match_question = results['documents'][0][0]  # The best-matching question
#         best_match_answer = results['metadatas'][0][0]['answer']  # Retrieve the corresponding answer
        
#         best_match_vector = embed_model.encode(best_match_question)
        
#         # Compute similarity
#         similarity_score = util.cos_sim(query_vector, best_match_vector).item()
        
#         print(f"Similarity Score: {similarity_score}")  # Debugging

#         # If similarity is below threshold, return None
#         if similarity_score < threshold:
#             return None  
        
#         return best_match_answer  # Return the correct FAQ answer

#     return None

# # ✅ Function to process user query & return AI-generated response
# def handle_user_query(user_input):
#     """Process user input and return AI-generated response."""
#     print("User:", user_input)

#     try:
#         # Retrieve relevant insurance info from ChromaDB
#         insurance_info = retrieve_insurance_info(user_input)
#         print("Insurance info retrieved:", insurance_info)

#         if insurance_info:
#             final_prompt = f"User: {user_input}\nInsurance Info: {insurance_info}\nAgent:"
#         else:
#             final_prompt = f"User: {user_input}\nAgent:"

#         # Generate AI response using Groq LLM
#         response = llm.invoke(prompt_template.format_messages(input=final_prompt))
#         print("AI response generated:", response.content)

#         # Store chat history
#         message_history.add_message(HumanMessage(content=user_input))
#         message_history.add_message(AIMessage(content=response.content))

#         return response.content  # Return response for Twilio
#     except Exception as e:
#         print("Error in handle_user_query:", str(e))
#         return "Sorry, I encountered an error while processing your request."

# # ✅ Flask app for Twilio voice response
# app = Flask(__name__)
# @app.route("/")
# def index():
#     return "Welcome to the AIphi Voice Service!"
# @app.route("/voice", methods=["POST"])
# def voice_response():
#     from twilio.twiml.voice_response import VoiceResponse
    
#     # Log the incoming request data for debugging
#     print("Incoming Twilio request data:", request.form)

#     # Get the user's speech input from Twilio's form data
#     user_input = request.form.get("SpeechResult", "Tell me about car insurance")  # Default if no input received
#     print("User input:", user_input)

#     try:
#         # Process the user input and generate a response
#         ai_response = handle_user_query(user_input)
#         print("AI response:", ai_response)

#         # Create Twilio voice response to speak back the AI's response
#         response = VoiceResponse()

#         # Continue the conversation until the user says goodbye
#         if "goodbye" in user_input.lower() or "end call" in user_input.lower():
#             response.say("Thank you for calling AIphi. Have a great day!", voice="Polly.Joanna", rate="80%")
#             response.hangup()  # End the call
#         else:
#             # If the user hasn't said goodbye, ask the user for another input
#             response.say(ai_response, voice="Polly.Joanna", rate="80%")
#             response.gather(input="speech", action="/voice")  # Keep listening for more speech input

#         return str(response)
#     except Exception as e:
#         # Log any errors that occur
#         print("Error in voice_response:", str(e))
#         response = VoiceResponse()
#         response.say("Sorry, an application error occurred. Goodbye!", voice="Polly.Joanna", rate="80%")
#         response.hangup()
#         return str(response)

# # ✅ Function to make a Twilio call
# def make_call(to_phone_number):
#     # Replace with your ngrok URL
#     local_server_url = "https://voluntary-genealogy-mh-catch.trycloudflare.com/voice"  # Example ngrok URL

#     # Corrected TwiML structure
#     twiml = f'''
#     <Response>
#         <Gather input="speech" action="{local_server_url}">
#             <Say voice="Polly.Joanna" rate="80%">Hello, this is Maya from AIphi. How can I help you today?</Say>
#         </Gather>
#     </Response>
#     '''

#     call = client_twilio.calls.create(
#         twiml=twiml,
#         to=to_phone_number,
#         from_=TWILIO_PHONE_NUMBER
#     )
#     print(f"Call initiated. Call SID: {call.sid}")

# def run_flask():
#     app.run(host="0.0.0.0", port=5009, debug=True, use_reloader=False)  # Set use_reloader=False to avoid double app run 

# # ✅ Streamlit Interface
# def streamlit_app():
#     st.title("AIphi Voice Service")
#     st.write("Enter the user's phone number to initiate a call and analyze sentiment.")

#     # Input phone number
#     phone_number = st.text_input("Enter the user's phone number (with country code):", "+12403554770")

#     # Button to initiate call
#     if st.button("Make Call"):
#         st.write(f"Calling {phone_number}...")
#         make_call(phone_number)
#         st.write("Call initiated. Please wait for the call to complete.")

#     # Sentiment analysis
#     if st.button("Analyze Sentiment"):
#         # Load sentiment analysis model
#         sentiment_pipeline = pipeline("sentiment-analysis")
        
#         # Example conversation (replace with actual call recording)
#         conversation = [
#             "I'm interested in your car insurance plans.",
#             "Great! Let me explain the benefits of our premium plan.",
#             "How much does it cost?",
#             "The premium plan costs $100 per month.",
#             "That sounds good. I'll think about it."
#         ]

#         # Analyze sentiment
#         sentiment_results = sentiment_pipeline(conversation)
#         st.write("Sentiment Analysis Results:")
#         for result in sentiment_results:
#             st.write(f"Text: {result['label']}, Score: {result['score']:.2f}")

# # ✅ Start Flask server and Streamlit app
# if __name__ == "__main__":
#     # Start Flask server in a separate thread
#     flask_thread = threading.Thread(target=run_flask)
#     flask_thread.start()

#     # Run Streamlit app
#     streamlit_app() 

import threading
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
import chromadb
from sentence_transformers import SentenceTransformer, util
from twilio.rest import Client
from flask import Flask, request
import os
import streamlit as st
from transformers import pipeline

# ✅ Twilio credentials
TWILIO_ACCOUNT_SID = "AC2780fbba0da5494f77fed33d3bd5ee9f"
TWILIO_AUTH_TOKEN = "49d878e96486dcdc30a98e9035019b81"  # Replace with your actual Auth Token
TWILIO_PHONE_NUMBER = "+17345304114"  # Replace with your Twilio Number

client_twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ✅ Initialize Groq LLM
groq_api_key = "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"
llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# ✅ Connect to Chroma DB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("geico_faqs")
 
# ✅ Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are Maya, a professional car insurance agent. Your job is to help users 
    understand car insurance options, guide them to the best plans, and explain benefits in simple terms."""),
    ("human", "{input}"),
])

# ✅ Initialize conversation memory
message_history = InMemoryChatMessageHistory()

# ✅ Global variable to store the conversation
conversation = []

# ✅ Function to fetch the best-matching insurance info from Chroma DB
def retrieve_insurance_info(user_query, threshold=0.4):
    query_vector = embed_model.encode(user_query)
    
    # Query ChromaDB to get top 1 best match
    results = collection.query(query_embeddings=[query_vector.tolist()], n_results=1)

    if results['documents']:
        best_match_question = results['documents'][0][0]  # The best-matching question
        best_match_answer = results['metadatas'][0][0]['answer']  # Retrieve the corresponding answer
        
        best_match_vector = embed_model.encode(best_match_question)
        
        # Compute similarity
        similarity_score = util.cos_sim(query_vector, best_match_vector).item()
        
        print(f"Similarity Score: {similarity_score}")  # Debugging

        # If similarity is below threshold, return None
        if similarity_score < threshold:
            return None  
        
        return best_match_answer  # Return the correct FAQ answer

    return None

# ✅ Function to process user query & return AI-generated response
def handle_user_query(user_input):
    """Process user input and return AI-generated response."""
    print("User:", user_input)

    try:
        # Retrieve relevant insurance info from ChromaDB
        insurance_info = retrieve_insurance_info(user_input)
        print("Insurance info retrieved:", insurance_info)

        if insurance_info:
            final_prompt = f"User: {user_input}\nInsurance Info: {insurance_info}\nAgent:"
        else:
            final_prompt = f"User: {user_input}\nAgent:"

        # Generate AI response using Groq LLM
        response = llm.invoke(prompt_template.format_messages(input=final_prompt))
        print("AI response generated:", response.content)

        # Store chat history
        message_history.add_message(HumanMessage(content=user_input))
        message_history.add_message(AIMessage(content=response.content))

        # Append to conversation
        conversation.append(f"User: {user_input}")
        conversation.append(f"Agent: {response.content}")

        return response.content  # Return response for Twilio
    except Exception as e:
        print("Error in handle_user_query:", str(e))
        return "Sorry, I encountered an error while processing your request."

# ✅ Flask app for Twilio voice response
app = Flask(__name__)
@app.route("/")
def index():
    return "Welcome to the AIphi Voice Service!"

@app.route("/voice", methods=["POST"])
def voice_response():
    from twilio.twiml.voice_response import VoiceResponse
    
    # Log the incoming request data for debugging
    print("Incoming Twilio request data:", request.form)

    # Get the user's speech input from Twilio's form data
    user_input = request.form.get("SpeechResult", "Tell me about car insurance")  # Default if no input received
    print("User input:", user_input)

    try:
        # Process the user input and generate a response
        ai_response = handle_user_query(user_input)
        print("AI response:", ai_response)

        # Create Twilio voice response to speak back the AI's response
        response = VoiceResponse()

        # Continue the conversation until the user says goodbye
        if "goodbye" in user_input.lower() or "end call" in user_input.lower():
            response.say("Thank you for calling AIphi. Have a great day!", voice="Polly.Joanna", rate="80%")
            response.hangup()  # End the call
        else:
            # If the user hasn't said goodbye, ask the user for another input
            response.say(ai_response, voice="Polly.Joanna", rate="80%")
            response.gather(input="speech", action="/voice")  # Keep listening for more speech input

        # Enable call recording and transcription
        response.record(timeout=10, transcribe=True, transcribeCallback="/transcribe")

        return str(response)
    except Exception as e:
        # Log any errors that occur
        print("Error in voice_response:", str(e))
        response = VoiceResponse()
        response.say("Sorry, an application error occurred. Goodbye!", voice="Polly.Joanna", rate="80%")
        response.hangup()
        return str(response)

@app.route("/transcribe", methods=["POST"])
def handle_transcription():
    # Get the transcription result from Twilio
    transcription_text = request.form.get("TranscriptionText", "")
    print("Transcription:", transcription_text)

    # Store the transcription in the conversation list
    conversation.append(f"Transcription: {transcription_text}")

    return "Transcription received", 200

# ✅ Function to make a Twilio call
def make_call(to_phone_number):
    # Replace with your ngrok URL
    local_server_url = "https://voluntary-genealogy-mh-catch.trycloudflare.com/voice"  # Example ngrok URL

    # Corrected TwiML structure
    twiml = f'''
    <Response>
        <Gather input="speech" action="{local_server_url}">
            <Say voice="Polly.Joanna" rate="80%">Hello, this is Maya from AIphi. How can I help you today?</Say>
        </Gather>
    </Response>
    '''

    call = client_twilio.calls.create(
        twiml=twiml,
        to=to_phone_number,
        from_=TWILIO_PHONE_NUMBER
    )
    print(f"Call initiated. Call SID: {call.sid}")

# ✅ Function to analyze sentiment and calculate verdict
def analyze_conversation(conversation):
    # Load sentiment analysis model
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    # Analyze each line of the conversation
    sentiment_scores = []
    for line in conversation:
        if line.startswith("User:"):
            result = sentiment_pipeline(line)
            sentiment_scores.append(result[0]['score'] if result[0]['label'] == 'POSITIVE' else 1 - result[0]['score'])

    # Calculate average sentiment score
    if sentiment_scores:
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        verdict = f"Based on the conversation, there is a {average_sentiment * 100:.2f}% chance the user will buy the insurance."
    else:
        verdict = "No user input detected in the conversation."

    return verdict
def run_flask():
    app.run(host="0.0.0.0", port=5009, debug=True, use_reloader=False)
# ✅ Streamlit Interface
def streamlit_app():
    st.title("AIphi Voice Service")
    st.write("Enter the user's phone number to initiate a call and analyze sentiment.")

    # Input phone number
    phone_number = st.text_input("Enter the user's phone number (with country code):")

    # Button to initiate call
    if st.button("Make Call"):
        st.write(f"Calling {phone_number}...")
        make_call(phone_number)
        st.write("Call initiated. Please wait for the call to complete.")

    # Display conversation
    if conversation:
        st.write("### Conversation Transcript")
        for line in conversation:
            st.write(line)

    # Analyze sentiment and provide verdict
    if st.button("Analyze Conversation"):
        verdict = analyze_conversation(conversation)
        st.write("### Verdict")
        st.write(verdict)

# ✅ Start Flask server and Streamlit app
if __name__ == "__main__":
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Run Streamlit app
    streamlit_app()