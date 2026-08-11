import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG with MistralAI", layout="wide")

st.title("Assistant Intelligent")
st.markdown("---")

# Initialisation de la session
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

# --- Helper functions HTTP ---
def fetch_conversations():
    try:
        res = requests.get(f"{API_URL}/conversations")
        return res.json() if res.status_code == 200 else []
    except Exception:
        return []

def load_conversation_messages(conv_id):
    res = requests.get(f"{API_URL}/conversations/{conv_id}/messages")
    if res.status_code != 200:
        st.error(f"Erreur chargement messages: {res.status_code} - {res.text}")
        return []
    return res.json()

def delete_conv(conv_id):
    try:
        requests.delete(f"{API_URL}/conversations/{conv_id}")
    except Exception:
        pass

# Sidebar : Gestion des Conversations 
st.sidebar.header("Conversations")

if st.sidebar.button("+ Nouvelle discussion", type="primary"):
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.rerun()

conversations = fetch_conversations()
for conv in conversations:
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        if st.button(conv["title"] or "Sans titre", key=f"conv_{conv['id']}"):
            st.session_state.conversation_id = conv["id"]
            st.session_state.messages = load_conversation_messages(conv["id"])
            st.rerun()
    with col2:
        if st.button("🗑", key=f"del_{conv['id']}"):
            delete_conv(conv["id"])
            if st.session_state.conversation_id == conv["id"]:
                st.session_state.conversation_id = None
                st.session_state.messages = []
            st.rerun()

st.sidebar.markdown("---")

# --- Sidebar : Gestion des fichiers et synchronisation ---
st.sidebar.header("Chargement")

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
    help="Copie le chemin depuis Finder (Clic droit > Copier comme nom de chemin). Sur Windows: C:\ → /mnt/c/ et les antislashs deviennent des slashs"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Sync Fichiers", type="secondary"):
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_names] if uploaded_files else []
        if new_files:
            with st.spinner("Synchronisation en cours..."):
                files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in new_files]
                resp = requests.post(f"{API_URL}/upload-multiple", files=files_payload)
                if resp.status_code == 200:
                    st.session_state.uploaded_names.extend([f.name for f in new_files])
                    sync_resp = requests.post(f"{API_URL}/sync")
                    if sync_resp.status_code == 200:
                        st.sidebar.success("Collection synchronisée !")
                    else:
                        st.sidebar.error("Erreur lors de la synchronisation.")
                else:
                    st.sidebar.error("Erreur lors de l'upload des fichiers.")
with col2:
    if st.button("Sync Répertoire", type="primary"):
        if not directory_path:
            st.sidebar.error("Veuillez entrer un chemin de répertoire valide.")
        elif not os.path.isdir(directory_path):
            st.sidebar.error(f"Le chemin '{directory_path}' n'existe pas.")
        else:
            with st.spinner("Synchronisation en cours..."):
                try:
                    resp = requests.post(f"{API_URL}/sync", json={"directory": directory_path})
                    if resp.status_code == 200:
                        st.sidebar.success("Répertoire synchronisé !")
                    else:
                        st.sidebar.error(f"Erreur API: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.sidebar.error(f"Erreur lors de la synchronisation du répertoire: {e}")

st.sidebar.markdown("---")
if st.sidebar.button(" Vider Collection", type="secondary"):
    with st.spinner("Vidage de la collection en cours..."):
        try:
            resp = requests.post(f"{API_URL}/clear-collection")
            if resp.status_code == 200:
                st.sidebar.success("Collection vidée complètement !")
                st.session_state.uploaded_names = []
            else:
                st.sidebar.error(f"Erreur lors du vidage: {resp.status_code} - {resp.text}")
        except Exception as e:
            st.sidebar.error(f"Erreur lors du vidage de la collection: {e}")

#Zone d'affichage des messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])
        if sources:
            with st.expander("Sources consultées"):
                for i, doc in enumerate(sources, start=1):
                    meta = doc.get("metadata", {}) or {}
                    source_path = meta.get("source", "Inconnu")
                    page = meta.get("page") or meta.get("page_number") or meta.get("page_num")
                    preview = doc.get("page_content", "")[:200]
                    st.markdown(f"**Source {i}:** `{source_path}`" + (f" (p.{page})" if page else ""))
                    st.markdown(f"> {preview}...")

#interaction user
if prompt := st.chat_input("Posez votre question..."):
    #if not active conversation, create one
    if st.session_state.conversation_id is None:
        title = prompt[:50] + "..." if len(prompt) > 50 else prompt
        try:
            res = requests.post(f"{API_URL}/conversations", json={"title": title})
            if res.status_code == 200:
                st.session_state.conversation_id = res.json().get("id")
            else:
                st.error(f"Impossible de créer une nouvelle conversation. Code d'erreur: {res.status_code}")
                st.stop()
        except Exception as e:
            st.error(f"Erreur lors de la création de la conversation: {e}")
            st.stop()


    # Add user message to the conversation
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send the question to the API and get the response
    with st.chat_message("assistant"):
        with st.spinner("Réfléchit..."):
            try:
                payload = {"query":prompt}
                if st.session_state.conversation_id:
                    payload["conversation_id"] = st.session_state.conversation_id

                response = requests.post(f"{API_URL}/ask", json=payload )
            
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

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
            except Exception as e:
                st.error(f"Erreur lors de la requête à l'API: {e}")