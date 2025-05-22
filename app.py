import streamlit as st
from backend import initialize_vector_store, query_chatbot, login_user, save_feedback
from pymongo import MongoClient
import os

# MongoDB setup
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["chatbot_db"]
chat_collection = db["chat_history"]

# Initialize vector store
vector_store = initialize_vector_store()

# Session state setup
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = []  # track feedback per message

# Login functionality
if not st.session_state.logged_in:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid username or password.")
else:
    st.title("📄 RAG PDF Chatbot")
    st.write(f"Logged in as: {st.session_state.username}")

    user_input = st.chat_input("Ask a question based on the documents...")

    if user_input:
        with st.spinner("Generating answer..."):
            # Get chatbot answer
            answer, context, citations = query_chatbot(
                vector_store, user_input, st.session_state.chat_history, username=st.session_state.username
            )

            # Update chat history
            st.session_state.chat_history.append({
                "user": user_input,
                "bot": answer,
                "context": context,
                "citations": citations
            })
            st.session_state.feedback_given.append(False)  # Track feedback

            # Insert into MongoDB
            chat_doc = {
                "username": st.session_state.username,
                "user_input": user_input,
                "bot_response": answer,
                "citations": citations
            }
            result = chat_collection.insert_one(chat_doc)
            print(f"✅ Inserted chat document with id: {result.inserted_id}")

    # Display chat messages with feedback
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["bot"])

            # Feedback section (only once per response)
            if not st.session_state.feedback_given[i]:
                with st.expander("💬 Provide Feedback", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        rating = st.selectbox("Rate this response:", [5, 4, 3, 2, 1], index=0)
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
