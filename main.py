import streamlit as st
import streamlit.components.v1 as components
import os
import json
import torch  # Import PyTorch pour vérifier la disponibilité de CUDA
from langchain_community.llms import Ollama
from document_loader import load_document_into_database
from models import get_list_of_models
from llm import getStreamingChain
import urllib

# Configurer la page Streamlit
st.set_page_config(layout="wide")  # Définit la page en mode large pour centrer le contenu

# Dossier pour stocker les comptes utilisateurs
ACCOUNTS_FOLDER = "accounts"
if not os.path.exists(ACCOUNTS_FOLDER):
    os.makedirs(ACCOUNTS_FOLDER)

# Dossier pour stocker les conversations
CONVERSATIONS_FOLDER = "conversations"
if not os.path.exists(CONVERSATIONS_FOLDER):
    os.makedirs(CONVERSATIONS_FOLDER)

# Charger les comptes existants
def load_accounts():
    accounts_file = os.path.join(ACCOUNTS_FOLDER, "accounts.json")
    if os.path.exists(accounts_file):
        with open(accounts_file, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # Retourner un dictionnaire vide si le fichier est vide ou corrompu
                return {}
    return {}

# Sauvegarder les comptes
def save_accounts(accounts):
    accounts_file = os.path.join(ACCOUNTS_FOLDER, "accounts.json")
    with open(accounts_file, "w") as f:
        json.dump(accounts, f)

# Créer un compte
def create_account(username, password):
    accounts = load_accounts()
    user_id = f"user_{len(accounts) + 1}"  # Générer l'ID utilisateur en fonction de l'ordre de création
    accounts[user_id] = {"username": username, "password": password}
    save_accounts(accounts)
    return user_id

# Vérifier les informations de connexion
def login(username, password):
    accounts = load_accounts()
    for user_id, info in accounts.items():
        if info["username"] == username and info["password"] == password:
            return user_id
    return None

# Charger les conversations sauvegardées
def load_conversations(user_id):
    conversations_path = os.path.join(CONVERSATIONS_FOLDER, f"{user_id}_conversations.json")
    if os.path.exists(conversations_path):
        with open(conversations_path, "r") as f:
            return json.load(f)
    return []

# Sauvegarder les conversations
def save_conversations(user_id, conversations):
    conversations_path = os.path.join(CONVERSATIONS_FOLDER, f"{user_id}_conversations.json")
    with open(conversations_path, "w") as f:
        json.dump(conversations, f)

# Sauvegarder une conversation spécifique
def save_specific_conversation(conversation):
    user_id = st.session_state["user_id"]
    save_conversations(user_id, st.session_state["conversations"])

# Supprimer une conversation spécifique
def delete_specific_conversation(index):
    del st.session_state["conversations"][index]
    save_conversations(st.session_state["user_id"], st.session_state["conversations"])
    st.experimental_rerun()

# Interface de connexion et création de compte
def login_page():
    st.title("Login / Création de Compte")

    # Sélectionner entre connexion et création de compte
    choice = st.selectbox("Choisissez une option", ["Se connecter", "Créer un compte"], key="login_choice")

    if choice == "Créer un compte":
        st.subheader("Créer un nouveau compte")
        username = st.text_input("Nom d'utilisateur", key="new_username")
        password = st.text_input("Mot de passe", type="password", key="new_password")

        if st.button("Créer un compte"):
            if username and password:
                user_id = create_account(username, password)
                st.success(f"Compte créé avec succès ! Votre ID utilisateur est : {user_id}")
                st.session_state["user_id"] = user_id  # Sauvegarde de l'utilisateur directement après création du compte
                st.experimental_rerun()  # Recharger la page pour entrer dans l'application
            else:
                st.error("Veuillez entrer un nom d'utilisateur et un mot de passe.")

    if choice == "Se connecter":
        st.subheader("Se connecter à un compte existant")
        username = st.text_input("Nom d'utilisateur", key="login_username")
        password = st.text_input("Mot de passe", type="password", key="login_password")

        if st.button("Se connecter"):
            user_id = login(username, password)
            if user_id:
                st.session_state["user_id"] = user_id  # Sauvegarde l'ID utilisateur dans la session
                st.success(f"Connexion réussie ! Bienvenue, {username}")
                st.experimental_rerun()  # Recharger la page pour afficher l'application principale
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")

    # Bouton de déconnexion juste en dessous
    if st.button("Se déconnecter"):
        st.session_state["user_id"] = None  # Réinitialiser l'ID utilisateur dans la session
        st.experimental_rerun()  # Recharger l'application pour revenir à la page de login

# Initialisation de la session utilisateur
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

# Vérifier la disponibilité de CUDA dès le démarrage
is_cuda_available = torch.cuda.is_available()

# Si l'utilisateur n'est pas connecté, afficher la page de login
if not st.session_state["user_id"]:
    login_page()  # Affiche la page de login
    st.stop()  # Arrête l'exécution ici jusqu'à ce que l'utilisateur soit connecté
else:
    # Charger le nom d'utilisateur à partir des comptes
    accounts = load_accounts()
    st.session_state["user_username"] = accounts[st.session_state["user_id"]]["username"]

    # Charger les conversations existantes
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = load_conversations(st.session_state["user_id"])

    # Charger une conversation sélectionnée
    if "selected_conversation" not in st.session_state:
        st.session_state["selected_conversation"] = []

    # Dossier personnel d'upload pour chaque utilisateur
    user_upload_folder = os.path.join("uploads", st.session_state["user_id"])
    if not os.path.exists(user_upload_folder):
        os.makedirs(user_upload_folder)

    # Charger la liste des modèles disponibles via Ollama (modèles d'embeddings et LLM)
    if "list_of_models" not in st.session_state:
        st.session_state["list_of_models"] = get_list_of_models()

    # Initialiser la clé "db" si elle n'existe pas déjà
    if "db" not in st.session_state:
        st.session_state["db"] = None

    # Charger l'historique de chat existant ou initialiser une nouvelle session de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialiser les paramètres s'ils n'existent pas déjà dans session_state
    if "model_params" not in st.session_state:
        st.session_state["model_params"] = {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "temperature": 0.7,
            "use_gpu": is_cuda_available and False  # Initialiser selon la disponibilité de CUDA
        }

    st.markdown("<h1 style='text-align: center;'>Local LLM with RAG 🤖</h1>", unsafe_allow_html=True)

    # Sidebar pour la gestion des documents et des conversations
    with st.sidebar:
        st.subheader("Options")

        # Section de gestion des documents
        with st.expander("Document"):
            selected_embedding_model = st.selectbox(
                "Choix de l'embeddeur pour l'indexation :",
                st.session_state["list_of_models"],
                key="select_embedding_model"
            )
            st.session_state["embedding_model"] = selected_embedding_model
            uploaded_files = st.file_uploader("Télécharger vos documents", accept_multiple_files=True)

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(user_upload_folder, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success(f"{len(uploaded_files)} fichiers téléchargés dans '{user_upload_folder}'.")

            st.write("### Documents disponibles :")
            files_in_folder = os.listdir(user_upload_folder)
            files_to_index = []

            if files_in_folder:
                # Fonction pour gérer le changement de la case "Sélectionner tout"
                def select_all_changed():
                    select_all = st.session_state['select_all_files']
                    for file_name in files_in_folder:
                        checkbox_key = f"select_{file_name}"
                        st.session_state[checkbox_key] = select_all

                # Fonction pour gérer le changement des cases individuelles
                def individual_checkbox_changed():
                    all_checked = all(st.session_state.get(f"select_{file_name}", False) for file_name in files_in_folder)
                    st.session_state['select_all_files'] = all_checked

                # Case à cocher "Sélectionner tout"
                select_all = st.checkbox("Sélectionner tout", key='select_all_files', on_change=select_all_changed)

                for file_name in files_in_folder:
                    col1, col2 = st.columns([9, 1])
                    with col1:
                        checkbox_key = f"select_{file_name}"
                        # Case à cocher individuelle
                        checked = st.checkbox(
                            f"{file_name}",
                            key=checkbox_key,
                            on_change=individual_checkbox_changed
                        )
                        if checked:
                            files_to_index.append(file_name)
                    with col2:
                        if st.button("🗑️", key=f"delete_{file_name}"):
                            os.remove(os.path.join(user_upload_folder, file_name))
                            st.experimental_rerun()

                # Case à cocher pour afficher/masquer les paramètres d'indexation
                show_index_params = st.checkbox("Afficher les paramètres d'indexation", key="show_index_params")

                if show_index_params:
                    st.write("### Paramètres pour l'indexation des documents")
                    chunk_size = st.slider(
                        "Chunk Size",
                        min_value=100,
                        max_value=2000,
                        value=st.session_state["model_params"]["chunk_size"],
                        step=50,
                        key="chunk_size_slider"
                    )
                    chunk_overlap = st.slider(
                        "Chunk Overlap",
                        min_value=0,
                        max_value=500,
                        value=st.session_state["model_params"]["chunk_overlap"],
                        step=10,
                        key="chunk_overlap_slider"
                    )

                    # Bouton pour sauvegarder les paramètres d'indexation
                    if st.button("Sauvegarder les paramètres d'indexation", key="save_index_params"):
                        st.session_state["model_params"]["chunk_size"] = chunk_size
                        st.session_state["model_params"]["chunk_overlap"] = chunk_overlap
                        st.success("Les paramètres d'indexation ont été sauvegardés.")

                if st.button("Indexer les documents"):
                    if files_to_index:
                        with st.spinner(f"🔍 Indexation des documents avec {selected_embedding_model}"):
                            for file in files_to_index:
                                try:
                                    st.session_state["db"] = load_document_into_database(
                                        st.session_state["embedding_model"],
                                        os.path.join(user_upload_folder, file),
                                        chunk_size=st.session_state["model_params"]["chunk_size"],
                                        chunk_overlap=st.session_state["model_params"]["chunk_overlap"]
                                    )
                                except ValueError as e:
                                    st.error(f"Erreur lors du chargement du fichier {file}: {e}")
                                except Exception as e:
                                    st.error(f"Erreur inattendue: {e}")
                        st.success("Documents indexés avec succès !")
                    else:
                        st.warning("Veuillez sélectionner au moins un document.")

        # Section de gestion des conversations enregistrées
        with st.expander("Conversations enregistrées"):
            st.write("### Conversations disponibles")
            for idx, conversation in enumerate(st.session_state["conversations"]):
                col1, col2, col3 = st.columns([8, 1, 1])
                with col1:
                    if st.button(f"Conversation {idx + 1}", key=f"load_conversation_{idx}_btn"):
                        st.session_state.messages = conversation
                        st.experimental_rerun()
                with col2:
                    if st.button("💾", key=f"save_conversation_{idx}_btn"):
                        save_specific_conversation(conversation)
                        st.success(f"Conversation {idx + 1} sauvegardée avec succès !")
                with col3:
                    if st.button("❌", key=f"delete_conversation_{idx}_btn"):
                        delete_specific_conversation(idx)
                        st.success(f"Conversation {idx + 1} supprimée avec succès !")

    st.markdown("<h2 style='text-align: center;'>Discussion avec le modèle LLM 🧠</h2>", unsafe_allow_html=True)

    # Sélection du modèle LLM sur la page principale
    selected_llm_model = st.selectbox("Choix du modèle LLM :", st.session_state["list_of_models"], key="select_llm_model")

    if "llm_model" not in st.session_state or st.session_state["llm_model"] != selected_llm_model:
        st.session_state["llm_model"] = selected_llm_model
        st.session_state["llm"] = Ollama(model=selected_llm_model, temperature=st.session_state["model_params"]["temperature"])

    # Paramètres du LLM
    with st.expander("Paramètres du LLM"):
        st.write("### Paramètres pour le modèle LLM")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state["model_params"]["temperature"],
            step=0.1
        )

        # Option pour utiliser le GPU
        use_gpu = st.checkbox(
            "Utiliser les processeurs graphiques (GPU)",
            value=st.session_state["model_params"]["use_gpu"],
            disabled=not is_cuda_available
        )

        # Bouton pour sauvegarder les paramètres du LLM
        if st.button("Sauvegarder les paramètres du LLM", key="save_llm_params"):
            st.session_state["model_params"]["temperature"] = temperature
            st.session_state["model_params"]["use_gpu"] = use_gpu
            st.success("Les paramètres du LLM ont été sauvegardés.")

    # Interface de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Posez une question basée sur les documents indexés"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.get("db") is None:
            st.warning("Veuillez indexer des documents avant de poser des questions.")
        else:
            with st.chat_message("assistant"):
                # Créer des colonnes pour le message de statut et le bouton "Stop"
                col_status, col_stop = st.columns([8, 1])
                with col_status:
                    status_placeholder = st.empty()
                    status_placeholder.markdown("Le modèle est en train d'écrire...")
                with col_stop:
                    if st.button("Stop"):
                        st.session_state['stop_generation'] = True

                response_placeholder = st.empty()

                try:
                    # Initialiser le drapeau 'stop_generation' s'il n'existe pas
                    if 'stop_generation' not in st.session_state:
                        st.session_state['stop_generation'] = False

                    # Démarrer la génération
                    stream = getStreamingChain(
                        prompt,
                        st.session_state.messages,
                        st.session_state["llm"],
                        st.session_state["db"],
                    )

                    response = ""
                    sources = ""
                    for chunk in stream:
                        # Vérifier si l'utilisateur a cliqué sur "Stop"
                        if st.session_state.get('stop_generation'):
                            break

                        if isinstance(chunk, dict):
                            response += chunk.get("answer", "")
                            sources = chunk.get("sources", "")
                        else:
                            response += chunk

                        # Mettre à jour le contenu de la réponse
                        response_placeholder.markdown(response)

                    # Vérifier si la génération a été interrompue par l'utilisateur
                    if st.session_state.get('stop_generation'):
                        # Afficher la notification dans le chat
                        response_placeholder.markdown("**Le modèle a été interrompu par l'utilisateur.**")
                        status_placeholder.empty()
                        st.session_state['stop_generation'] = False
                    else:
                        # Supprimer le message de statut une fois la génération terminée
                        status_placeholder.empty()
                        # Ajouter la réponse aux messages de la session
                        response_with_sources = f"{response}\n\n**Sources:** {sources}" if sources else response
                        st.session_state.messages.append({"role": "assistant", "content": response_with_sources})
                        st.session_state["conversations"].append(st.session_state.messages)
                        save_conversations(st.session_state["user_id"], st.session_state["conversations"])

                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de la génération de la réponse : {e}")
