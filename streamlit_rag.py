import streamlit as st

st.title("🔎 LangChain - Chat with automatic Search")

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Create input fields for question and related link
question = st.text_input("What is your question?")
web_link = st.text_input("Provide a related link")

# Check if both inputs are provided
if question and web_link:
    # Append the user's question to messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # Append the user's link to messages
    st.session_state.messages.append({"role": "link", "content": web_link})
    st.chat_message("link").write(web_link)