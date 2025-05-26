import streamlit as st
from backend import initialize_vector_store, query_chatbot, login_user, save_feedback
from pymongo import MongoClient
import os
import uuid
import datetime
import pandas as pd  # Import pandas for CSV export

# ---------------- Utility Functions ----------------
def generate_session_id():
    return str(uuid.uuid4())

def format_timestamp(ts):
    return ts.strftime("%Y-%m-%d %H:%M")

def is_greeting(text):
    greetings = ["hi", "hello", "hey", "hii", "hi there", "hello there", "hloo", "good morning", "good evening"]
    text_clean = text.lower().strip().rstrip("!?.")
    return any(greet == text_clean for greet in greetings)

# ---------------- MongoDB Setup ----------------
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["chatbot_db"]
chat_collection = db["chat_history"]

# ---------------- Vector Store Init ----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = initialize_vector_store()

# ---------------- Session State Init ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None  # Set on login or new chat
if "session_first_query" not in st.session_state:
    st.session_state.session_first_query = None

# ---------------- Update Last Bot Response in DB ----------------
def update_last_chat_response(username, user_query, new_bot_response):
    chat_doc = chat_collection.find_one(
        {"username": username, "user_input": user_query},
        sort=[("_id", -1)]
    )
    if chat_doc:
        chat_collection.update_one(
            {"_id": chat_doc["_id"]},
            {"$set": {"bot_response": new_bot_response}}
        )

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.title("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.session_id = generate_session_id()
            st.session_state.chat_history = []
            st.session_state.feedback_given = []
            st.session_state.session_first_query = None
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

# ---------------- MAIN CHAT UI ----------------
else:
    # Sidebar with Chat Sessions
    with st.sidebar:
        st.title("📚 Chat Sessions")

        # New Chat button resets history & session ID
        if st.button("➕ New Chat"):
            st.session_state.chat_history = []
            st.session_state.feedback_given = []
            st.session_state.session_id = generate_session_id()
            st.session_state.session_first_query = None
            st.rerun()

        st.markdown("### 📜 Chat History")

        # Aggregate sessions for current user
        pipeline = [
            {"$match": {"username": st.session_state.username}},
            {"$match": {"session_first_query": {"$exists": True, "$ne": ""}}},  # Exclude invalid sessions
            {"$sort": {"timestamp": 1}},
            {"$group": {
                "_id": "$session_id",
                "first_query": {"$first": "$session_first_query"},
                "timestamp": {"$first": "$timestamp"}
            }}
        ]
        session_docs = list(chat_collection.aggregate(pipeline))

        for doc in session_docs:
            sid = doc["_id"]
            first_query = doc.get("first_query", "").strip()
            label = first_query.capitalize()
            label = label[:47] + "..." if len(label) > 50 else label

            col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
            with col1:
                if st.button(f"🗂️ {label}", key=f"load_{sid}"):
                    chats = list(chat_collection.find(
                        {"username": st.session_state.username, "session_id": sid}
                    ).sort("timestamp", 1))

                    st.session_state.chat_history = [
                        {
                            "user": c["user_input"],
                            "bot": c["bot_response"],
                            "context": c.get("context", ""),
                            "citations": c.get("citations", []),
                            "timestamp": c.get("timestamp", datetime.datetime.utcnow())
                        }
                        for c in chats
                    ]
                    st.session_state.feedback_given = [True for _ in st.session_state.chat_history]
                    st.session_state.session_id = sid
                    st.session_state.session_first_query = first_query
                    st.rerun()

            with col2:
                # Add a download button for each session
                chats = list(chat_collection.find(
                    {"username": st.session_state.username, "session_id": sid}
                ).sort("timestamp", 1))
                if chats:
                    chat_history_df = pd.DataFrame([
                        {
                            "User": c["user_input"],
                            "Bot": c["bot_response"],
                            "Citations": c.get("citations", []),
                            "Timestamp": c.get("timestamp", datetime.datetime.utcnow())
                        }
                        for c in chats
                    ])
                    chat_history_csv = chat_history_df.to_csv(index=False)
                    st.download_button(
                        label="📥",
                        data=chat_history_csv,
                        file_name=f"{st.session_state.username}_session_{sid}.csv",
                        mime="text/csv",
                        key=f"download_{sid}"
                    )

            with col3:
                if st.button("🗑️", key=f"delete_{sid}"):
                    chat_collection.delete_many({
                        "username": st.session_state.username,
                        "session_id": sid
                    })
                    st.success("🗑️ Chat deleted!")
                    st.rerun()

    st.title("📄 RAG PDF Chatbot")
    st.write(f"Logged in as: {st.session_state.username}")

    # Display chat messages with timestamps
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(chat["user"])

        with st.chat_message("assistant"):
            st.markdown(chat["bot"])
            
            # Feedback UI if feedback not given for this message
            if not st.session_state.feedback_given[i]:
                with st.expander("💬 Provide Feedback", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        rating = st.selectbox("Rate this response:", [5, 4, 3, 2, 1], index=0, key=f"rating_{i}")
                    with col2:
                        comments = st.text_input("Any comments?", key=f"comment_{i}")
                    if st.button("Submit Feedback", key=f"submit_{i}"):
                        save_feedback(
                            username=st.session_state.username,
                            user_query=chat["user"],
                            bot_response=chat["bot"],
                            rating=rating,
                            comments=comments if comments else None
                        )
                        st.success("✅ Feedback submitted!")
                        st.session_state.feedback_given[i] = True
                        st.rerun()

    # Regenerate last response if last user query is not a greeting
    if st.session_state.chat_history:
        last_chat = st.session_state.chat_history[-1]
        if not is_greeting(last_chat["user"].strip()):
            if st.button("🔁 Regenerate Response", key="regen_last"):
                with st.spinner("Regenerating response..."):
                    new_answer, context, citations = query_chatbot(
                        st.session_state.vector_store,
                        last_chat["user"],
                        st.session_state.chat_history[:-1],
                        username=st.session_state.username
                    )
                    # Update state and DB
                    st.session_state.chat_history[-1]["bot"] = new_answer
                    update_last_chat_response(st.session_state.username, last_chat["user"], new_answer)
                    st.rerun()

    # Chat input box
    user_input = st.chat_input("Ask a question based on the documents...")
    if user_input:
        with st.spinner("Generating answer..."):
            answer, context, citations = query_chatbot(
                st.session_state.vector_store,
                user_input,
                st.session_state.chat_history,
                username=st.session_state.username
            )

            # Append new chat message
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": answer,
                "context": context,
                "citations": citations,
                "timestamp": datetime.datetime.utcnow()
            })
            st.session_state.feedback_given.append(False)

            # Track if first query in this session
            is_first = len(st.session_state.chat_history) == 1

            chat_doc = {
                "username": st.session_state.username,
                "user_input": user_input,
                "bot_response": answer,
                "citations": citations,
                "context": context,
                "session_id": st.session_state.session_id,
                "timestamp": datetime.datetime.utcnow()
            }
            if is_first or st.session_state.session_first_query is None:
                chat_doc["session_first_query"] = user_input
                st.session_state.session_first_query = user_input

            # Insert chat into MongoDB
            chat_collection.insert_one(chat_doc)
            st.rerun()
