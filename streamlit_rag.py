from openai import OpenAI

import streamlit as st
import re


with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="langchain_search_api_key_openai", type="password"
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"


st.title("🔎 LangChain - Auto Search")


# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]


# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)
    st.chat_message("user").write(prompt)

    #augment the prompt
    prompt += "Please provide relevant links that can be helpful for me"
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(model="gpt-4", messages=st.session_state.messages)
    msg = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
    st.chat_message("assistant").write(response)

    urls = re.findall(r'https?://[^\s\)]+', msg)

    st.write(urls)
    print(urls)

    #st.markdown("![Alt Text](https://www.python.org/)")
    #st.image("https://www.python.org/")
