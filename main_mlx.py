# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
# and streamlit chat example (https://streamlit.io/generative-ai)

import streamlit as st
import random
import time

# import client
from chatter_mlx import Chatter

# MODEL = "mlx-community/CodeLlama-70b-Instruct-hf-4bit-MLX"
# MODEL = "mlx-community/phi-2"
# MODEL = "mlx-community/Mixtral-8x7B-Instruct-v0.1"
# MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4-bit"
# MODEL = "mlx-community/NeuralBeagle14-7B-mlx"
MODEL = "mlx-community/NeuralBeagle14-7B-4bit-mlx"

@st.cache_resource
def Initialize_model(model_id):
    return Chatter(model_id)

chatbot = Initialize_model(MODEL)

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with {MODEL.capitalize()}")

with col2:
    st.button('Reset ↺', on_click=reset_chat)
    max_tokens = st.slider('Max Tokens', min_value=1,
                           max_value=5000, value=500)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        context = st.session_state.context

        response, ctx = chatbot.chat(prompt, context, max_tokens=max_tokens)

        for chunk in response:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        st.session_state.context = ctx

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})
