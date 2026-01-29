import streamlit as st
import requests
st.set_page_config(page_title="RAG avec MistralAI", layout="wide")

st.title("Assistant PDF Intelligent")
st.markdown("---")

#gestion du pdf
st.sidebar.header("Chargement")
uploaded_file=st.sidebar.file_uploader("Choisissez un document PDF", type="pdf")

if uploaded_file:
    if st.sidebar.button("Indexer le document"):
        with st.spinner("Analyse du PDF en cours..."):
            # On envoie le fichier à l'API FastAPI
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post("http://127.0.0.1:8000/upload", files=files)
            
            if response.status_code == 200:
                st.sidebar.success("Document prêt !")
            else:
                st.sidebar.error("Erreur lors de l'indexation.")

# 2. Interface de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if prompt := st.chat_input("Posez votre question sur le document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mistral réfléchit..."):
            # Appel à l'API FastAPI pour poser la question
            response = requests.post("http://127.0.0.1:8000/ask", json={"query": prompt})
            
            if response.status_code == 200:
                answer = response.json()["reponse"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("L'API n'a pas pu répondre. Vérifiez si le PDF est bien indexé.")