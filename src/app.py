import streamlit as st
import requests
import os
import time
from streamlit_cookies_controller import CookieController

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Assistant Intelligent", layout="wide")

controller = CookieController()

#Sidebar style
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        background-color: #f7f7f8;
    }
    section[data-testid="stSidebar"] button {
        border-radius: 8px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-weight: 400 !important;
    }
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #e8e8ea !important;
        color: #1a1a1a !important;
        border: none !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        border: none !important;
        color: #3a3a3a !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #ececee !important;
    }
    div[data-testid="stSidebarUserContent"] hr {
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Session State
if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.user = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []
if "page" not in st.session_state:
    st.session_state.page = "Chat"

def auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}

#Auth - login/register/cookie
def login(email, password):
    try:
        response = requests.post(f"{API_URL}/auth/login", json={"email": email, "password": password})
        if response.status_code != 200:
            return False
        token = response.json()["access_token"]
        st.session_state.token = token
        me = requests.get(f"{API_URL}/auth/me", headers=auth_headers())
        if me.status_code != 200:
            st.session_state.token = None
            return False
        st.session_state.user = me.json()
    except Exception:
        return False

    try:
        controller.set("auth_token", token)
    except Exception:
        pass 

    return True

def register(email, password):
    try:
        res = requests.post(f"{API_URL}/auth/register", json={"email": email, "password": password})
    except Exception as e:
        return False, f"Erreur réseau: {e}"

    if res.status_code != 200:
        return False, res.json().get("detail", "Erreur inconnue")

    token = res.json()["access_token"]
    st.session_state.token = token

    me = requests.get(f"{API_URL}/auth/me", headers=auth_headers())
    if me.status_code != 200:
        st.session_state.token = None
        return False, f"Compte créé mais erreur de connexion automatique: {me.text}"

    st.session_state.user = me.json()

    try:
        controller.set("auth_token", token)
    except Exception:
        pass

    return True, None

def try_restore_session():
    """Retrieves the token stored on navigation part if streamlit session lost."""
    if st.session_state.token is not None:
        return

    if "cookie_check_attempts" not in st.session_state:
        st.session_state.cookie_check_attempts = 0

    try:
        cached_token = controller.get("auth_token")
    except TypeError:
        cached_token = None

    if cached_token:
        st.session_state.token = cached_token
        me = requests.get(f"{API_URL}/auth/me", headers=auth_headers())
        if me.status_code == 200:
            st.session_state.user = me.json()
        else:
            st.session_state.token = None
            controller.remove("auth_token")

    if st.session_state.cookie_check_attempts < 2:
        st.session_state.cookie_check_attempts += 1
        time.sleep(0.3)
        st.rerun()

def logout():
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.conversation_id = None
    st.session_state.messages = []
    controller.remove("auth_token")


try_restore_session()

if st.session_state.token is None:
    st.title("Connexion")
    tab_login, tab_register = st.tabs(["Connexion", "Créer un compte"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Mot de passe", type="password", key="login_password")
            submitted = st.form_submit_button("Connexion")
            if submitted:
                if login(email, password):
                    st.rerun()
                else:
                    st.error("Email ou mot de passe incorrect")

    with tab_register:
        with st.form("register_form"):
            email_r = st.text_input("Email", key="register_email")
            password_r = st.text_input("Mot de passe", type="password", key="register_password")
            created = st.form_submit_button("Créer un compte")
            if created:
                ok, error = register(email_r, password_r)
                if ok:
                    st.rerun()
                else:
                    st.error(f"Erreur lors de la création du compte: {error}")

    st.stop()  # Stop further execution until the user is logged in

#Helper http (conversations/documents)
def fetch_conversations():
    try:
        res = requests.get(f"{API_URL}/conversations", headers=auth_headers())
        return res.json() if res.status_code == 200 else []
    except Exception:
        return []

def load_conversation_messages(conv_id):
    res = requests.get(f"{API_URL}/conversations/{conv_id}/messages", headers=auth_headers())
    if res.status_code != 200:
        st.error(f"Erreur chargement messages: {res.status_code} - {res.text}")
        return []
    return res.json()

def delete_conv(conv_id):
    try:
        requests.delete(f"{API_URL}/conversations/{conv_id}", headers=auth_headers())
    except Exception:
        pass

# Sidebar navigation
with st.sidebar:
    st.markdown(f"**{st.session_state.user['email']}**")
    st.divider()

    nav_options = ["🏠︎ Home", "💬︎ Chat", "📄︎ Documents"]
    if st.session_state.user.get("is_admin"):
        nav_options.append("⚙︎ Admin")

    for opt in nav_options:
        label = opt.split(" ", 1)[1]
        is_active = st.session_state.page == label
        if st.button(opt, key=f"nav_{label}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = label
            st.rerun()

    st.divider()

    if st.session_state.page == "Chat":
        if st.button("＋ Nouvelle discussion", use_container_width=True, type="secondary"):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

        st.caption("Historique")
        for conv in fetch_conversations():
            col1, col2 = st.columns([5, 1])
            with col1:
                is_active = st.session_state.conversation_id == conv["id"]
                if st.button(
                    conv["title"] or "Sans titre",
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
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

    st.divider()
    if st.button("Déconnexion", use_container_width=True):
        logout()
        st.rerun()

#Page Home
if st.session_state.page == "Home":
    st.title("Assistant Intelligent")
    st.markdown("Bienvenue. Utilise le menu à gauche pour discuter avec tes documents ou en ingérer de nouveaux.")

#Page Documents (ingestion)
elif st.session_state.page == "Documents":
    st.title("Gestion des documents")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_files = st.file_uploader(
        "Choisissez des documents (glisse un dossier entier ou plusieurs fichiers)",
        type=["pdf", "txt", "pptx", "xlsx", "csv", "docx"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_files:
        col_count, col_clear = st.columns([3, 1])
        with col_count:
            st.caption(f"{len(uploaded_files)} fichier(s) sélectionné(s)")
        with col_clear:
            if st.button("Tout effacer"):
                st.session_state.uploader_key += 1
                st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Synchroniser les fichiers", type="secondary", use_container_width=True):
            new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_names] if uploaded_files else []
            if new_files:
                with st.spinner("Synchronisation en cours..."):
                    files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in new_files]
                    resp = requests.post(f"{API_URL}/upload-multiple", files=files_payload, headers=auth_headers())
                    if resp.status_code == 200:
                        st.session_state.uploaded_names.extend([f.name for f in new_files])
                        sync_resp = requests.post(f"{API_URL}/sync", headers=auth_headers())
                        if sync_resp.status_code == 200:
                            st.success("Collection synchronisée !")
                        else:
                            st.error("Erreur lors de la synchronisation.")
                    else:
                        st.error("Erreur lors de l'upload des fichiers.")

    with col2:
        if st.button("Vider ma collection", type="secondary", use_container_width=True):
            with st.spinner("Vidage en cours..."):
                try:
                    resp = requests.post(f"{API_URL}/clear-collection", headers=auth_headers())
                    if resp.status_code == 200:
                        st.success("Vos documents ont été supprimés !")
                        st.session_state.uploaded_names = []
                    else:
                        st.error(f"Erreur: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.error(f"Erreur: {e}")

    with st.expander("Options avancées (usage développeur)"):
        directory_path = st.text_input(
            "Chemin serveur à synchroniser :",
            placeholder="data/mon-dossier"
        )
        if st.button("Synchroniser ce dossier serveur"):
            if directory_path:
                with st.spinner("Synchronisation en cours..."):
                    resp = requests.post(f"{API_URL}/sync", json={"directory": directory_path}, headers=auth_headers())
                    if resp.status_code == 200:
                        st.success("Dossier synchronisé !")
                    else:
                        st.error(f"Erreur: {resp.status_code} - {resp.text}")
            else:
                st.warning("Précise un chemin de dossier.")

#Page Admin
elif st.session_state.page == "Admin":
    st.title("Admin")
    #soon

#Page Chat
elif st.session_state.page == "Chat":
    st.title("Assistant Intelligent")

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

    if prompt := st.chat_input("Posez votre question..."):
        if st.session_state.conversation_id is None:
            title = prompt[:50] + "..." if len(prompt) > 50 else prompt
            res = requests.post(f"{API_URL}/conversations", json={"title": title}, headers=auth_headers())
            if res.status_code == 200:
                st.session_state.conversation_id = res.json().get("id")
            else:
                st.error(f"Impossible de créer une nouvelle conversation. Code: {res.status_code}")
                st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Réfléchit..."):
                try:
                    payload = {"query": prompt, "conversation_id": st.session_state.conversation_id}
                    response = requests.post(f"{API_URL}/ask", json=payload, headers=auth_headers())

                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

                        if sources:
                            with st.expander("Sources consultées"):
                                for i, doc in enumerate(sources, start=1):
                                    meta = doc.get("metadata", {}) or {}
                                    source_path = meta.get("source", "Inconnu")
                                    page = meta.get("page") or meta.get("page_number") or meta.get("page_num")
                                    preview = doc.get("page_content", "")[:200]
                                    st.markdown(f"**Source {i}:** `{source_path}`" + (f" (p.{page})" if page else ""))
                                    st.markdown(f"> {preview}...")
                    else:
                        st.error("L'API n'a pas pu répondre.")
                except Exception as e:
                    st.error(f"Erreur lors de la requête à l'API: {e}")