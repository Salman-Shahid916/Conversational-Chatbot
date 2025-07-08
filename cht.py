import streamlit as st  # Streamlit for UI
from dotenv import load_dotenv  # Load .env into os.environ
import os  # Interacts with environment vars
from langchain_groq import ChatGroq  # Groq LLM integration
from langchain.memory import ConversationBufferMemory  # Memory backend for chat
from langchain.chains import ConversationChain  # Chain that wires LLM + memory

# Load API key
load_dotenv()  # Read .env file
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Set GROQ API key

# Streamlit App setup
st.set_page_config(page_title="Conversational Chatbot")  # Title in browser tab
st.title("Conversational Chatbot With Message History")  # App header

# Sidebar controls
model_name = st.sidebar.selectbox(
    "Select Groq Model",
    ["gemma2-9b-it", "deepseek-r1-distill-llama-70b", "llama3-8b-8192"]
)

temperature = st.sidebar.slider(
    "Temperature", 0.0, 1.0, 0.7
)

max_tokens = st.sidebar.slider(
    "Max Tokens", 50, 300, 150
)

# Initialize memory and history
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True
    )

if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.chat_input("You:")  # Clears itself on enter

if user_input:
    # Append user message
    st.session_state.history.append(("user", user_input))

    # Instantiate a fresh LLM
    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Build conversation chain with memory
    conv = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=True
    )

    # Get AI response (memory updates internally)
    ai_response = conv.predict(input=user_input)

    # Append assistant message
    st.session_state.history.append(("assistant", ai_response))

# Render chat messages
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)
