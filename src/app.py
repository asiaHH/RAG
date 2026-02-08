import streamlit as st
import requests
import os

st.set_page_config(page_title="RAG with MistralAI", layout="wide")

st.title("Assistant Intelligent")
st.markdown("---")

st.sidebar.header("Chargement")

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

uploaded_files = st.sidebar.file_uploader(
    "Choisissez des documents",
    type=["pdf", "txt", "pptx", "xlsx", "csv"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
directory_path = st.sidebar.text_input(
    "Chemin du répertoire :",
    value="data",
    placeholder="/Users/...",
    help="Copie le chemin depuis Finder (Clic droit > Copier comme nom de chemin)"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("Sync Fichiers", type="secondary"):
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_names] if uploaded_files else []
        if new_files:
            with st.spinner("Synchronisation en cours..."):
                files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in new_files]
                resp = requests.post("http://127.0.0.1:8000/upload-multiple", files=files_payload)
                if resp.status_code == 200:
                    st.session_state.uploaded_names.extend([f.name for f in new_files])
                    sync_resp = requests.post("http://127.0.0.1:8000/sync")
                    if sync_resp.status_code == 200:
                        st.sidebar.success("Collection synchronisée !")
                    else:
                        st.sidebar.error("Erreur lors de la synchronisation.")
                else:
                    st.sidebar.error("Erreur lors de l'upload des fichiers.")
with col2:
    if st.sidebar.button("Sync Répertoire", type="primary"):
        if directory_path and os.path.isdir(directory_path):
            with st.spinner("Synchronisation en cours..."):
                resp = requests.post("http://127.0.0.1:8000/sync")
                if resp.status_code == 200:
                    st.sidebar.success("Répertoire synchronisé !")
                else:
                    st.sidebar.error("Erreur lors de la synchronisation du répertoire.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Réfléchit..."):
            response = requests.post("http://127.0.0.1:8000/ask", json={"query": prompt})
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                if sources:
                    st.markdown("**Sources :**")
                    for i, doc in enumerate(sources, start=1):
                        meta = doc.get("metadata", {}) or {}
                        source_path = meta.get("source", "Inconnu")
                        page = meta.get("page") or meta.get("page_number") or meta.get("page_num") or meta.get("pagenumber") 
                        try:
                            if page is not None:
                                page = int(page)
                                if page == 0:
                                    page = page + 1
                        except Exception:
                            page = meta.get("page")

                        preview = doc.get("page_content", "")[:200]

                        with st.expander(f"Source {i}: {source_path}" + (f" (p.{page})" if page else "")):
                            st.markdown(f"**Extrait:** {preview}...")
                            st.markdown(f"**Chemin:** `{source_path}`")
                            if page:
                                st.markdown(f"**Page:** {page}")

            else:
                st.error("L'API n'a pas pu répondre. Vérifiez si le PDF est bien indexé.")