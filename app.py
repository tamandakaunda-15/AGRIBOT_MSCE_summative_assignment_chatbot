import streamlit as st
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from datasets import Dataset
import os
import uuid 

# --- CONFIGURATION & CONSTANTS ---
MODEL_HUB_ID = "TamandaKaunda/MSCE-Agriculture-T5" 
MAX_LENGTH = 200


@st.cache_resource
def load_model_and_tokenizer():
    try:
        # 1. Retrieve the token securely from the st.secrets dictionary
        # This key (HUGGING_FACE_TOKEN) must match the key in your secrets.toml file.
        HF_AUTH_TOKEN = st.secrets["HUGGING_FACE_TOKEN"]
        
        # 2. Use the token when calling the from_pretrained functions
        # This tells the model loader to use your credentials to download the private/large files.
        tokenizer = T5Tokenizer.from_pretrained(
            MODEL_HUB_ID, 
            token=HF_AUTH_TOKEN
        )
        model = TFT5ForConditionalGeneration.from_pretrained(
            MODEL_HUB_ID, 
            token=HF_AUTH_TOKEN
        )
        return tokenizer, model
    except Exception as e:
        # Authentication failure occurs here if the token is wrong or the file is not found
        st.error(f"Authentication Failed. Ensure HUGGING_FACE_TOKEN is set in secrets.toml and the model path is correct. Details: {e}")
        st.stop()

# --- 0. SESSION STATE INITIALIZATION ---

if 'qa_messages' not in st.session_state:
    st.session_state.qa_messages = [{"role": "assistant", "content": "Welcome! Ask me a question about soil degradation, livestock, or farm mechanization."}]

if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "QA_CHAT" 

# --- Callbacks ---
def switch_mode(mode):
    st.session_state.app_mode = mode

def clear_chat_history():
    st.session_state.qa_messages = [{"role": "assistant", "content": "Conversation cleared. Ready for new questions."}]
    st.rerun()
    
# --- 1. MODEL AND DATA CACHING (Final Logic) ---
@st.cache_resource
def load_hub_model_and_tokenizer(): # Renamed function for clarity
    try:
        # 1. Retrieve the token securely from the st.secrets dictionary
        HF_AUTH_TOKEN = st.secrets["HUGGING_FACE_TOKEN"]
        
        # 2. Use the token when calling the from_pretrained functions
        tokenizer = T5Tokenizer.from_pretrained(
            MODEL_HUB_ID, 
            token=HF_AUTH_TOKEN
        )
        model = TFT5ForConditionalGeneration.from_pretrained(
            MODEL_HUB_ID, 
            token=HF_AUTH_TOKEN
        )
        return tokenizer, model
    except Exception as e:
        # This handles authentication failure when deploying to Streamlit Cloud
        st.error(f"Authentication Failed. Check secrets.toml and Hub ID. Details: {e}")
        st.stop()


# --- 0. SESSION STATE INITIALIZATION (Remainder of the app) ---
# ... (All other functions and logic remain the same) ...

# --- 1. MODEL AND DATA CACHING (Final Logic) ---
# *** THIS SECTION MUST BE DELETED AND REPLACED WITH THE CALLS BELOW ***

@st.cache_data
def load_full_dataset():
    try:
        qa_df = pd.read_json('msce_agriculture_qa.json')
        dataset = qa_df.to_dict('records')
        return dataset
    except FileNotFoundError:
        st.error("Error: msce_agriculture_qa.json not found for context lookup.")
        return []

# Load resources (Calling the function defined at the top)
tokenizer, model = load_hub_model_and_tokenizer() # Call the correct function
full_dataset = load_full_dataset()


# --- 2. CORE LOGIC FUNCTIONS (Final Logic) ---

def get_context_for_generation(user_question, data):
    user_q_tokens = set(user_question.lower().strip().split())
    best_match = None
    max_score = 0
    
    stop_words = {'what', 'is', 'a', 'the', 'of', 'and', 'or', 'do', 'does', 'are', 'in', 'can', '?', 'types', 're', 'define', 'agoforstry', 'agroforestry', 'livestock'}

    significant_user_tokens = user_q_tokens - stop_words
    
    if not data:
        return user_question

    for example in data:
        dataset_q_tokens = set(example['question'].lower().strip().split())
        shared_keywords = significant_user_tokens.intersection(dataset_q_tokens)
        score = len(shared_keywords)
        
        if score > max_score:
            max_score = score
            best_match = example['context']

    if max_score >= 2: 
        return best_match
    else:
        return user_question 

def generate_answer(question, context):
    if context == question:
        input_text = f"question: {question} context: I can only answer questions related to the MSCE Agriculture syllabus. Please ask a specific question based on the provided materials."
    else:
        input_text = f"question: {question} context: {context}"
        
    input_ids = tokenizer(input_text, return_tensors="tf", padding=True, truncation=True, max_length=512).input_ids
    output = model.generate(
        input_ids,
        max_length=MAX_LENGTH,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_question(answer_text, model, tokenizer):
    input_text = f"generate question: {answer_text}"
    input_ids = tokenizer(input_text, return_tensors="tf", max_length=128, truncation=True).input_ids
    output = model.generate(
        input_ids,
        max_length=64, 
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# --- FUNCTION TO LOAD EXTERNAL CSS ---
def load_css(file_name):
    """Reads a CSS file and applies it to the Streamlit app."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at {file_name}. Default styling applied.")


# =========================================================================
# === STREAMLIT UI: SETUP AND SIDEBAR =====================================
# =========================================================================


# --- SIDEBAR NAVIGATION (Hamburger Menu) ---
with st.sidebar:
    st.header("MSCE Agro-Bot")
    st.markdown("---")

    # --- Student Mode Button ---
    if st.button("Student Mode (Q&A)", use_container_width=True, key="switch_qa"):
        switch_mode("QA_CHAT")
        
    # --- Tutor Mode Button ---
    if st.button("Tutor Mode (Q-Gen)", use_container_width=True, key="switch_qg"):
        switch_mode("QG_MODE")
        
    st.markdown("---")
    
    # --- Delete/Clear Conversation Button ---
    if st.button("Clear Chat History", use_container_width=True, key="clear_chat", on_click=clear_chat_history, help="Delete all messages in the current session"):
        pass


# =========================================================================
# === MAIN APP FLOW CONTROL ===============================================
# =========================================================================

st.title("MSCE Agriculture Chatbot")
st.caption("Generative AI Assistant for the Malawi Secondary Certificate of Education Syllabus.")

if st.session_state.app_mode == "QA_CHAT":
    
    # =========================================================================
    # === 1. QA CHAT MODE (Primary Student Interface) ==========================
    # =========================================================================
    st.header("Question & Answer Mode")
    
    # Display chat messages from history
    for message in st.session_state.qa_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter your agriculture question here...", key="qa_input_main"):
        
        st.session_state.qa_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if full_dataset:
            with st.spinner('Retrieving context and generating answer...'):
                context = get_context_for_generation(prompt, full_dataset)
                response = generate_answer(prompt, context)

                with st.chat_message("assistant"):
                    st.markdown(response)
                
                st.session_state.qa_messages.append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                st.markdown("I'm sorry, the knowledge base is unavailable.")


elif st.session_state.app_mode == "QG_MODE":
    
    # =========================================================================
    # === 2. QUESTION GENERATION MODE (Tutor/Teacher Interface) ================
    # =========================================================================
    st.header("Question Generation Tutor")
    st.info("Paste a key answer or syllabus paragraph below. The AI will generate a relevant question for testing.")
    
    # Text area for user to input a block of text
    answer_input = st.text_area(
        "Source Text / Key Answer:", 
        key="qg_area_main",
        value="Land drainage is the practice of removing excess moisture from the soil to make it suitable for growing crops.",
        height=200
    )

    if st.button("Generate Question", key="qg_button_main"):
        if len(answer_input.split()) < 5:
             st.warning("Please enter a longer text snippet (at least 5 words) to generate a meaningful question.")
        else:
            with st.spinner('Analyzing text and generating question...'):
                generated_q = generate_question(answer_input, model, tokenizer)
                
                st.success("Generated Practice Question:")
                st.markdown(f"### {generated_q}")
