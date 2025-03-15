
# from langchain.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from tts_stt import transcribe_audio, generate_speech
# import chromadb
# from sentence_transformers import SentenceTransformer

# # Initialize Groq LLM
# groq_api_key = "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"
# llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# # ✅ Connect to Chroma DB (Persistent Storage)
# client = chromadb.PersistentClient(path="./chroma_db")
# collection = client.get_collection("geico_faqs")

# # Load embedding model
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Define prompt template
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", """You are Maya, a professional car insurance agent. Your job is to help users 
#     understand car insurance options, guide them to the best plans, and explain benefits in simple terms."""),  
#     ("human", "{input}"),
# ])

# # Initialize conversation memory
# message_history = InMemoryChatMessageHistory()

# # ✅ Function to fetch best-matching insurance information from ChromaDB
# def retrieve_insurance_info(user_query):
#     query_vector = embed_model.encode(user_query).tolist()

#     # Query ChromaDB to get the top 1 best match
#     results = collection.query(query_embeddings=[query_vector], n_results=1)

#     if results['documents']:
#         best_match_question = results['documents'][0][0]  # Retrieved FAQ question
#         best_match_answer = results['metadatas'][0][0]["answer"]  # Retrieved corresponding answer
#         return f"Q: {best_match_question}\nA: {best_match_answer}"
    
#     return "I'm sorry, but I couldn't find relevant information. Let me guide you based on my knowledge."

# # ✅ Function to handle the call
# def handle_call():
#     greeting = "Hello, this is Maya from AIphi. How can I assist you today?"
#     print("Agent:", greeting)
#     generate_speech(greeting, "greeting.wav")

#     while True:
#         print("Listening for user input...")
#         user_audio_path = "output.wav"

#         # Transcribe user's speech
#         user_input = transcribe_audio(user_audio_path)
#         print("User:", user_input)

#         # Retrieve relevant insurance info from ChromaDB
#         insurance_info = retrieve_insurance_info(user_input)

#         # Format the final prompt
#         prompt = f"User: {user_input}\nInsurance Info: {insurance_info}\nAgent:"

#         # Generate AI response using Groq LLM
#         response = llm.invoke(prompt_template.format_messages(input=prompt))

#         # Store chat history
#         message_history.add_message(HumanMessage(content=user_input))
#         message_history.add_message(AIMessage(content=response.content))

#         # Speak the response
#         print("Agent:", response.content)
#         generate_speech(response.content, "response.wav")

#         if "goodbye" in user_input.lower() or "end call" in user_input.lower():
#             farewell = "Thank you for calling AIphi. Have a great day!"
#             print("Agent:", farewell)
#             generate_speech(farewell, "end_call.wav")
#             break
# from langchain.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from tts_stt import transcribe_audio, generate_speech
# import chromadb
# from sentence_transformers import SentenceTransformer, util

# # Initialize Groq LLM
# groq_api_key = "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"
# llm = ChatGroq(api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# # Connect to Chroma DB
# client = chromadb.PersistentClient(path="./chroma_db")
# collection = client.get_collection("geico_faqs")

# # Load embedding model
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Define prompt template
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", """You are Maya, a professional car insurance agent. Your job is to help users 
#     understand car insurance options, guide them to the best plans, and explain benefits in simple terms."""),
#     ("human", "{input}"),
# ])

# # Initialize conversation memory
# message_history = InMemoryChatMessageHistory()

# # Function to fetch the best-matching insurance information from Chroma DB with similarity check
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

# # Function to handle the call
# def handle_call():
#     greeting = "Hello, this is Maya from AIphi. How can I assist you today?"
#     print("Agent:", greeting)
#     generate_speech(greeting, "greeting.wav")

#     while True:
#         print("Listening for user input...")
#         user_audio_path = "output.wav"
        
#         # Transcribe user's speech
#         user_input = transcribe_audio(user_audio_path)
#         print("User:", user_input)

#         # Retrieve relevant insurance info from Chroma DB
#         insurance_info = retrieve_insurance_info(user_input)

#         if insurance_info:
#             final_prompt = f"User: {user_input}\nInsurance Info: {insurance_info}\nAgent:"
#         else:
#             final_prompt = f"User: {user_input}\nAgent:"

#         # Generate AI response using Groq LLM
#         response = llm.invoke(prompt_template.format_messages(input=final_prompt))
        
#         # Store chat history
#         message_history.add_message(HumanMessage(content=user_input))
#         message_history.add_message(AIMessage(content=response.content))

#         # Speak the response
#         print("Agent:", response.content)
#         generate_speech(response.content, "response.wav")

#         if "goodbye" in user_input.lower() or "end call" in user_input.lower():
#             farewell = "Thank you for calling AIphi. Have a great day!"
#             print("Agent:", farewell)
#             generate_speech(farewell, "end_call.wav")
#             break
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

#     # Retrieve relevant insurance info from ChromaDB
#     insurance_info = retrieve_insurance_info(user_input)

#     if insurance_info:
#         final_prompt = f"User: {user_input}\nInsurance Info: {insurance_info}\nAgent:"
#     else:
#         final_prompt = f"User: {user_input}\nAgent:"

#     # Generate AI response using Groq LLM
#     response = llm.invoke(prompt_template.format_messages(input=final_prompt))

#     # Store chat history
#     message_history.add_message(HumanMessage(content=user_input))
#     message_history.add_message(AIMessage(content=response.content))

#     print("Agent:", response.content)
#     return response.content  # Return response for Twilio

# # ✅ Flask app for Twilio voice response
# app = Flask(__name__)
# @app.route("/")
# def index():
#     return "Welcome to the AIphi Voice Service!"
# @app.route("/voice", methods=["POST"])
# def voice_response():
#     from twilio.twiml.voice_response import VoiceResponse
    
#     # Get the user's speech input from Twilio's form data
#     user_input = request.form.get("SpeechResult", "Tell me about car insurance")  # Default if no input received

#     # Process the user input and generate a response
#     ai_response = handle_user_query(user_input)

#     # Create Twilio voice response to speak back the AI's response
#     response = VoiceResponse()

#     # Continue the conversation until the user says goodbye
#     if "goodbye" in user_input.lower() or "end call" in user_input.lower():
#         response.say("Thank you for calling AIphi. Have a great day!", voice="alice")
#         response.hangup()  # End the call
#     else:
#         # If the user hasn't said goodbye, ask the user for another input
#         response.say(ai_response, voice="alice")
#         response.gather(input="speech", action="/voice")  # Keep listening for more speech input

#     return str(response)

# # ✅ Function to make a Twilio call
# def make_call(to_phone_number):
#     local_server_url = "https://photography-nine-works-quantitative.trycloudflare.com/voice"
#   # Local server URL

#     call = client_twilio.calls.create(
#         twiml=f'<Response><Gather input="speech" action="{local_server_url}"/><Say>Hello, this is Maya from AIphi. How can I help you today?</Say></Gather></Response>',
#         to=to_phone_number,
#         from_=TWILIO_PHONE_NUMBER
#     )
#     print(f"Call initiated. Call SID: {call.sid}") 

# def run_flask():
#     app.run(host="0.0.0.0",port=5009, debug=True, use_reloader=False)  # , ssl_context=('cert.pem', 'key.pem') Set `use_reloader=False` to avoid double app run 

# # ✅ Start Flask server and make a call
# if __name__ == "__main__":
#     # Start Flask server in a separate thread
#     flask_thread = threading.Thread(target=run_flask)
#     flask_thread.start()

#     phone_number_to_call = "+12403554770"  # Change this to the actual phone number you want to call
#     make_call(phone_number_to_call) 
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

    # Retrieve relevant insurance info from ChromaDB
    insurance_info = retrieve_insurance_info(user_input)

    if insurance_info:
        final_prompt = f"User: {user_input}\nInsurance Info: {insurance_info}\nAgent:"
    else:
        final_prompt = f"User: {user_input}\nAgent:"

    # Generate AI response using Groq LLM
    response = llm.invoke(prompt_template.format_messages(input=final_prompt))

    # Store chat history
    message_history.add_message(HumanMessage(content=user_input))
    message_history.add_message(AIMessage(content=response.content))

    print("Agent:", response.content)
    return response.content  # Return response for Twilio

# ✅ Flask app for Twilio voice response
app = Flask(__name__)
@app.route("/")
def index():
    return "Welcome to the AIphi Voice Service!"
@app.route("/voice", methods=["POST"])
def voice_response():
    from twilio.twiml.voice_response import VoiceResponse
    
    # Get the user's speech input from Twilio's form data
    user_input = request.form.get("SpeechResult", "Tell me about car insurance")  # Default if no input received

    # Process the user input and generate a response
    ai_response = handle_user_query(user_input)

    # Create Twilio voice response to speak back the AI's response
    response = VoiceResponse()

    # Continue the conversation until the user says goodbye
    if "goodbye" in user_input.lower() or "end call" in user_input.lower():
        response.say("Thank you for calling AIphi. Have a great day!", voice="alice")
        response.hangup()  # End the call
    else:
        # If the user hasn't said goodbye, ask the user for another input
        response.say(ai_response, voice="alice")
        response.gather(input="speech", action="/voice")  # Keep listening for more speech input

    return str(response)

# ✅ Function to make a Twilio call
def make_call(to_phone_number):
    local_server_url = "https://voluntary-genealogy-mh-catch.trycloudflare.com/voice"  # Local server URL

    # Corrected TwiML structure
    twiml = f'''
    <Response>
        <Gather input="speech" action="{local_server_url}">
            <Say>Hello, this is Maya from AIphi. How can I help you today?</Say>
        </Gather>
    </Response>
    '''

    call = client_twilio.calls.create(
        twiml=twiml,
        to=to_phone_number,
        from_=TWILIO_PHONE_NUMBER
    )
    print(f"Call initiated. Call SID: {call.sid}")

def run_flask():
    app.run(host="0.0.0.0",port=5009, debug=True, use_reloader=False)  # , ssl_context=('cert.pem', 'key.pem') Set use_reloader=False to avoid double app run 

# ✅ Start Flask server and make a call
if __name__ == "__main__":
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    phone_number_to_call = "+12403554770"  # Change this to the actual phone number you want to call
    make_call(phone_number_to_call)