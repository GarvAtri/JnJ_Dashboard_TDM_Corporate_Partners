import streamlit as st
import pandas as pd
import os
import google.generativeai as genai

# Configure page
st.set_page_config(page_title="LLM Powered Data Dashboard", layout="wide")

# App title
st.title("📊 LLM Powered Data Dashboard")
st.markdown("Welcome to the interactive data dashboard for the **Manasi and Pragathi** project.")

# Optional: Set up Gemini API Key from environment or input
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Upload Data", "Explore and Analyze", "Start a New Chat"])
    
    st.divider()
    st.header("Settings")
    api_key = st.text_input("Gemini API Key", value=st.session_state.gemini_api_key, type="password")
    if api_key:
        st.session_state.gemini_api_key = api_key
        try:
            genai.configure(api_key=api_key)
        except Exception:
            # configure may raise if SDK not initialized yet; let requests fail later with clearer error
            pass
        # Model selection / listing
        st.markdown("---")
        model_name = st.text_input("Model name", value=st.session_state.model_name, help="Enter a model name (e.g. a Gemini model). If unsure, click 'List available models'.")
        if model_name:
            st.session_state.model_name = model_name

        if st.button("List available models"):
            if not st.session_state.gemini_api_key:
                st.error("Please add your Gemini API key first so we can call ListModels.")
            else:
                try:
                    models = []
                    for m in genai.list_models():
                        name = None
                        if isinstance(m, dict):
                            name = m.get('name') or m.get('model') or str(m)
                        else:
                            name = getattr(m, 'name', None) or getattr(m, 'model', None) or str(m)
                        models.append(name)
                    if models:
                        sel = st.selectbox("Available models", models)
                        st.session_state.model_name = sel
                        st.success(f"Selected model: {sel}")
                    else:
                        st.info("No models returned by the API.")
                except Exception as e:
                    st.error(f"Error listing models: {e}")

# Session state initialization for chat and data
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "model_name" not in st.session_state:
    st.session_state.model_name = ""

# --- Page: Upload Data ---
if page == "Upload Data":
    st.header("Upload Your Data")
    st.write("Upload a CSV or Excel file to get started with exploration and analysis.")
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("File successfully uploaded!")
            st.write("### Data Preview")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

# --- Page: Explore and Analyze ---
elif page == "Explore and Analyze":
    st.header("Explore and Analyze Data")
    if st.session_state.df is not None:
        df = st.session_state.df
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        
        st.write("### Full Dataset")
        st.dataframe(df, use_container_width=True)
        
        st.write("### Summary Statistics")
        st.write(df.describe(include='all'))
        
        st.write("### Ask the LLM about this data")
        if not st.session_state.gemini_api_key:
            st.warning("Please enter your Gemini API key in the sidebar to use the AI analysis feature.")
        else:
            analysis_prompt = st.text_area("What do you want to know about this dataset?", placeholder="E.g., What are the key trends in this dataset?")
            if st.button("Analyze Data"):
                with st.spinner("Analyzing..."):
                    try:
                        # Convert a sample of the data to text for the LLM
                        data_sample = df.head(50).to_csv(index=False)
                        model_name = st.session_state.model_name
                        if not model_name:
                            st.error("Please select a model in the sidebar (use 'List available models' or type a model name).")
                        else:
                            model = genai.GenerativeModel(model_name)
                            response = model.generate_content(
                                f"You are a data analyst. Here is a sample of the dataset:\n{data_sample}\n\nUser Question: {analysis_prompt}"
                            )
                        # SDK returns an object with a text attribute in many versions; protectively handle fallback
                        output_text = getattr(response, 'text', None) or str(response)
                        st.write(output_text)
                    except Exception as e:
                        st.error(f"Error communicating with Gemini: {e}")
    else:
        st.info("No data available. Please go to the 'Upload Data' tab to upload a file.")

# --- Page: Start a New Chat ---
elif page == "Start a New Chat":
    st.header("Chat with the Assistant")
    
    if not st.session_state.gemini_api_key:
        st.warning("Please enter your Gemini API key in the sidebar to chat with the LLM.")
    else:
        col1, col2 = st.columns([0.8, 0.2])
        with col2:
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
                
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Thinking..."):
                try:
                    model_name = st.session_state.model_name
                    if not model_name:
                        st.error("Please select a model in the sidebar (use 'List available models' or type a model name).")
                    else:
                        model = genai.GenerativeModel(model_name)

                        # Create context including data schema if available
                        context = ""
                        if st.session_state.df is not None:
                            context = f"\n\nContext: You have access to a dataset with columns: {list(st.session_state.df.columns)}. The user might ask questions related to this."
                        
                        response = model.generate_content(prompt + context)
                        
                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
                            assistant_text = getattr(response, 'text', None) or str(response)
                            st.markdown(assistant_text)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                except Exception as e:
                    st.error(f"Error: {e}")
