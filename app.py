import streamlit as st
import random
import time
from streamlit_js_eval import streamlit_js_eval
from extraction_agent import create_extraction_agent,read_products
from langchain_community.callbacks import get_openai_callback


with st.sidebar:
        st.subheader("Configuración")
        if st.button("Reiniciar"):
            st.session_state.main_agent = create_extraction_agent()
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
            

st.title("Jamón y Salud")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.extraction_agent = create_extraction_agent()
    

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Introduce tu pedido: "):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        with st.spinner("Procesando ..."):

            result = st.session_state.extraction_agent.invoke({"input":prompt})
            st.write(result)
            

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result})