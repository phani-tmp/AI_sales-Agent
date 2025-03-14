
# import chromadb
# from sentence_transformers import SentenceTransformer
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# # Initialize Selenium WebDriver
# service = Service("chromedriver.exe")  # Ensure you have ChromeDriver installed
# driver = webdriver.Chrome(service=service)

# # Open the insurance website
# url = "https://www.geico.com/information/aboutinsurance/auto/faq/"  # Replace with actual URL
# driver.get(url)

# # Wait for the page to load
# wait = WebDriverWait(driver, 10)

# # Locate all FAQ questions
# faq_items = driver.find_elements(By.XPATH, '//*[@id="faq-accordion"]/li/div[1]')  # Clickable questions

# faq_data = {}

# for index, faq in enumerate(faq_items, start=1):
#     question = faq.text.strip()

#     # Click the FAQ question to reveal answer
#     driver.execute_script("arguments[0].click();", faq)

#     # Wait for the answer element using the modified XPath
#     answer_xpath = f'//*[@id="faq-accordion"]/li[{index}]/div[2]/div/p'
    
#     try:
#         answer_element = wait.until(
#             EC.visibility_of_element_located((By.XPATH, answer_xpath))
#         )
#         answer = answer_element.text.strip()
#     except Exception as e:
#         answer = "Answer not found."

#     # Store question-answer pair
#     faq_data[question] = answer

# # Print extracted FAQs
# for question, answer in faq_data.items():
#     print(f"Q: {question}\nA: {answer}\n")

# # Close the browser
# driver.quit()

# # Load embedding model
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize Chroma client (using new client initialization)
# client = chromadb.Client()

# # Create a collection to store FAQ data
# collection_name = "geico_faqs"
# collection = client.create_collection(collection_name)

# # Convert FAQs to embeddings and insert them into Chroma DB
# for question, answer in faq_data.items():
#     # Generate embedding for the question
#     vector = embed_model.encode(question).tolist()
    
#     # Insert the question and answer into the Chroma collection
#     collection.add(
#         documents=[question],
#         metadatas=[{"answer": answer}],
#         embeddings=[vector],
#         ids=[question]
#     )

# print("FAQs successfully stored in Chroma DB!")
# # Assuming this is your data extraction script
# import chromadb

# client = chromadb.Client()

# # Create collection if it doesn't exist
# collection_name = "geico_faqs"
# if collection_name not in client.list_collections():
#     collection = client.create_collection(collection_name)
# else:
#     collection = client.get_collection(collection_name)
# import chromadb
# from sentence_transformers import SentenceTransformer
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.action_chains import ActionChains
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# # Initialize Selenium WebDriver
# service = Service("chromedriver.exe")  # Ensure you have ChromeDriver installed
# driver = webdriver.Chrome(service=service)

# # Open the insurance website
# url = "https://www.geico.com/information/aboutinsurance/auto/faq/"  # Replace with actual URL
# driver.get(url)

# # Wait for the page to load
# wait = WebDriverWait(driver, 10)

# # Locate all FAQ questions
# faq_items = driver.find_elements(By.XPATH, '//*[@id="faq-accordion"]/li/div[1]')  # Clickable questions

# faq_data = {}

# for index, faq in enumerate(faq_items, start=1):
#     question = faq.text.strip()

#     # Click the FAQ question to reveal answer
#     driver.execute_script("arguments[0].click();", faq)

#     # Wait for the answer element using the modified XPath
#     answer_xpath = f'//*[@id="faq-accordion"]/li[{index}]/div[2]/div/p'
    
#     try:
#         answer_element = wait.until(
#             EC.visibility_of_element_located((By.XPATH, answer_xpath))
#         )
#         answer = answer_element.text.strip()
#     except Exception as e:
#         answer = "Answer not found."

#     # Store question-answer pair
#     faq_data[question] = answer

# # Print extracted FAQs
# for question, answer in faq_data.items():
#     print(f"Q: {question}\nA: {answer}\n")

# # Close the browser
# driver.quit()

# # Load embedding model
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize Chroma client (using new client initialization)
# client = chromadb.Client()

# # Check if collection exists, if not, create it
# collection_name = "geico_faqs"
# if collection_name not in client.list_collections():
#     collection = client.create_collection(collection_name)
# else:
#     collection = client.get_collection(collection_name)

# # Convert FAQs to embeddings and insert them into Chroma DB
# for question, answer in faq_data.items():
#     # Generate embedding for the question
#     vector = embed_model.encode(question).tolist()
    
#     # Insert the question and answer into the Chroma collection
#     collection.add(
#         documents=[question],
#         metadatas=[{"answer": answer}],
#         embeddings=[vector],
#         ids=[question]
#     )

# print("FAQs successfully stored in Chroma DB!")
# collections = client.list_collections()
# print("Available collections:", collections)
import chromadb
from sentence_transformers import SentenceTransformer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize Selenium WebDriver
service = Service("chromedriver.exe")  # Ensure you have ChromeDriver installed
driver = webdriver.Chrome(service=service)

# Open the insurance website
url = "https://www.geico.com/information/aboutinsurance/auto/faq/"  # Replace with actual URL
driver.get(url)

# Wait for the page to load
wait = WebDriverWait(driver, 10)

# Locate all FAQ questions
faq_items = driver.find_elements(By.XPATH, '//*[@id="faq-accordion"]/li/div[1]')  # Clickable questions

faq_data = {}

for index, faq in enumerate(faq_items, start=1):
    question = faq.text.strip()

    # Click the FAQ question to reveal answer
    driver.execute_script("arguments[0].click();", faq)

    # Wait for the answer element using the modified XPath
    answer_xpath = f'//*[@id="faq-accordion"]/li[{index}]/div[2]/div/p'
    
    try:
        answer_element = wait.until(
            EC.visibility_of_element_located((By.XPATH, answer_xpath))
        )
        answer = answer_element.text.strip()
    except Exception as e:
        answer = "Answer not found."

    # Store question-answer pair
    faq_data[question] = answer

# Print extracted FAQs
for question, answer in faq_data.items():
    print(f"Q: {question}\nA: {answer}\n")

# Close the browser
driver.quit()

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Chroma client with persistent storage in the current directory
client = chromadb.PersistentClient(path="./chroma_db")  # Save in the current directory

# Create a collection to store FAQ data
collection_name = "geico_faqs"
collection = client.get_or_create_collection(collection_name)

# Convert FAQs to embeddings and insert them into Chroma DB
for question, answer in faq_data.items():
    # Generate embedding for the question
    vector = embed_model.encode(question).tolist()
    
    # Insert the question and answer into the Chroma collection
    collection.add(
        documents=[question],
        metadatas=[{"answer": answer}],
        embeddings=[vector],
        ids=[question]
    )

print("FAQs successfully stored in Chroma DB!")