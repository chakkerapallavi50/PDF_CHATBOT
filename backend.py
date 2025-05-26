from vector_store import VectorStore
from pdf_utils import process_all_pdfs
import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# MongoDB Setup
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["chatbot_db"]
users_collection = db["users"]
chat_collection = db["chat_history"]
feedback_collection = db["chat_feedback"]

# Load users from JSON
with open("users.json") as f:
    users = json.load(f)
    for user in users:
        if not users_collection.find_one({"username": user["username"]}):
            users_collection.insert_one(user)

# Authenticate user
def login_user(username, password):
    user = users_collection.find_one({"username": username, "password": password})
    return user is not None

# Initialize vector store
def initialize_vector_store():
    vector_store = VectorStore("data/")
    if not vector_store.is_index_built():
        print("Ingesting and indexing PDFs...")
        chunks = process_all_pdfs("data/")
        vector_store.build_index(chunks)
        print("Indexing completed.")
    return vector_store

# Format citations
def format_citations(relevant_chunks):
    citations_dict = {}
    for chunk in relevant_chunks:
        source = chunk.get("source", "Unknown source")
        page = chunk.get("page", "Unknown page")
        if source not in citations_dict:
            citations_dict[source] = set()
        citations_dict[source].add(str(page))

    formatted_citations = []
    for source, pages in citations_dict.items():
        sorted_pages = sorted(pages, key=lambda x: (x == "Unknown page", x))
        if len(sorted_pages) == 1:
            formatted_citations.append(f"{source} (page {sorted_pages[0]})")
        else:
            pages_str = ", ".join(sorted_pages)
            formatted_citations.append(f"{source} (pages {pages_str})")

    return "\n".join(formatted_citations)

# Greeting set
greetings = {"hi", "hello", "hey", "hii", "good morning", "good evening"}

def is_greeting(text):
    return text.strip().lower() in greetings

# Save chat history
def save_chat_history(username, user_query, bot_response):
    chat_doc = {
        "username": username,
        "query": user_query,
        "response": bot_response,
        "timestamp": datetime.utcnow()
    }
    result = chat_collection.insert_one(chat_doc)
    print(f"Inserted chat document with id: {result.inserted_id}")

# Update chat response
def update_last_chat_response(username, user_query, new_response, new_context=None, new_citations=None):
    filter_query = {
        "username": username,
        "query": user_query
    }
    update_fields = {
        "response": new_response,
        "timestamp": datetime.utcnow(),
    }
    if new_context is not None:
        update_fields["context"] = new_context
    if new_citations is not None:
        update_fields["citations"] = new_citations

    result = chat_collection.update_one(filter_query, {"$set": update_fields})
    if result.matched_count == 0:
        print(f"⚠️ No matching chat found to update for user '{username}' and query '{user_query}'")
    else:
        print(f"✅ Updated chat response for user '{username}' and query '{user_query}'")

# Save feedback
def save_feedback(username, user_query, bot_response, rating, comments=None):
    feedback_doc = {
        "username": username,
        "query": user_query,
        "response": bot_response,
        "rating": rating,
        "comments": comments,
        "timestamp": datetime.utcnow()
    }
    result = feedback_collection.insert_one(feedback_doc)
    print(f"Inserted feedback with id: {result.inserted_id}")

# Query chatbot
def query_chatbot(vector_store, user_query, chat_history, username=None):
    is_greet = is_greeting(user_query)
    relevant_chunks = [] if is_greet else vector_store.search(user_query, k=7)
    
    context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
    formatted_history = "\n".join(
        [f"User: {entry['user']}\nBot: {entry['bot']}" for entry in chat_history]
    )

    prompt = f"""
You are a professional AI assistant. Respond appropriately to the user's query.
- If the user greets (e.g., "hi", "hello", "hey"), respond with a friendly greeting and do not include citations.
- If the user asks a question based on the documents, use the provided context to answer.
- If the context is insufficient, say: "I'm sorry, I couldn't find relevant information in the documents."

Context:
{context}

Conversation History:
{formatted_history}

User Query: {user_query}

Answer:
""".strip()

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    answer = response.text.strip()

    is_fallback = "i'm sorry, i couldn't find relevant information" in answer.lower()

    # Attach citations only if it's not greeting or fallback and relevant chunks exist
    if not (is_greet or is_fallback) and relevant_chunks:
        citations_text = format_citations(relevant_chunks)
        answer += f"\n\n📚 Citations:\n{citations_text}"

    if username:
        save_chat_history(username, user_query, answer)

    return answer, context, relevant_chunks
