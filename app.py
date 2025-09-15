import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from deep_translator import GoogleTranslator
import sqlite3
from datetime import datetime
import re
import os

# ============================
# Initialisation base SQLite
# ============================
def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            summary TEXT,
            topic TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Enregistrement dans l'historique

def log_interaction(question, answer, summary, topic):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO history VALUES (?, ?, ?, ?, ?)",
                   (datetime.now().isoformat(), question, answer, summary, topic))
    conn.commit()
    conn.close()

# === Détection langue + traduction
def detect_and_translate(text, target_lang="en"):
    keywords_fr = ["qu'est-ce", "définis", "c'est quoi", "en quoi", "définition", "algorithme", "réseau", "python", "fonction", "logique"]
    if any(k in text.lower() for k in keywords_fr):
        return GoogleTranslator(source='auto', target=target_lang).translate(text), 'fr'
    else:
        return text, 'en'
    
# Retraduction de la réponse en français si nécessaire
def retranslate_if_needed(answer, original_lang):
    if original_lang == 'fr':
        return GoogleTranslator(source='en', target='fr').translate(answer)
    return answer

def detect_topic(question):
    topics = {
        "programmation": ["python", "code", "boucle", "fonction"],
        "maths": ["algèbre", "matrice", "nombre", "logique"],
        "algorithmique": ["algorithme", "complexité", "tri", "pseudo-code"],
        "Reseau": ["reseau","bit"]
    }
    for topic, keywords in topics.items():
        if any(k in question.lower() for k in keywords):
            return topic
    return "autre"

# Nettoyage basique du texte généré

def clean_output(text):
    text = re.sub(r"[^a-zA-Z0-9 à-ü,.!?()\-]", '', text)
    return text.strip().capitalize()

# Résumé automatique d’un texte via Flan-T5

def generate_summary(text, pipe):
    prompt = f"Résume ce texte en français : {text}"
    return pipe(prompt, max_length=100, do_sample=False)[0]['generated_text']

# === Nettoyage simple
def clean_output(text):
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9 ,.'\"()!?-]", '', text)
    return text.strip().capitalize()

# === Chargement du modèle local
with st.spinner("Chargement du modèle FLAN-T5..."):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
    local_llm = HuggingFacePipeline(pipeline=pipe)

# === Chargement FAISS & embeddings
index_path = "index/faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local(index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever()

# === Interface utilisateur Streamlit
st.set_page_config(page_title="Assistant Pédagogique IA", layout="wide")
st.title("🎓 Assistant pédagogique IA")
st.write("Pose ta question sur un cours 📚")
init_db()

# Champ de saisie utilisateur

question = st.text_input("💬 Ta question ici :")

if question:
    with st.spinner("Recherche et génération..."):
        question_en, lang = detect_and_translate(question, target_lang="en")
        topic = detect_topic(question)


        qa_chain_local = RetrievalQA.from_chain_type(llm=local_llm, retriever=retriever, return_source_documents=True)
        result = qa_chain_local({"query": question_en})
        answer = clean_output(result["result"])
        
        # Fallback si réponse trop courte ou insatisfaisante
        if len(answer.split()) < 5 or answer.lower() in ["internet", "python", "bit"]:
            st.warning("Réponse locale trop courte. Appel à OpenAI GPT-3.5 en secours...")
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    openai_llm = OpenAI(temperature=0.3, max_tokens=300, openai_api_key=openai_key, model_name="gpt-3.5-turbo")
                    qa_chain_gpt = RetrievalQA.from_chain_type(llm=openai_llm, retriever=retriever, return_source_documents=True)
                    result = qa_chain_gpt({"query": question})
                    answer = result["result"]
                except Exception as e:
                    st.error(f"Erreur OpenAI : {e}")
            else:
                st.error("❌ Clé OpenAI manquante. Ajoutez-la à `.env` ou aux variables d’environnement.")

#  Retraduction si nécessaire
        answer = retranslate_if_needed(answer, lang)
        summary = generate_summary(answer, pipe)

# Affichage des résultats
        st.markdown(f"\n💡 **Réponse :** {answer}")
        st.markdown(f"📝 **Résumé :** {summary}")
        st.markdown(f"📚 **Thème détecté :** {topic}")

# Enregistrement dans l’historique
        log_interaction(question, answer, summary, topic)

# Affichage des sources utilisées
        with st.expander("📄 Sources utilisées :"):
            for doc in result["source_documents"]:
                st.write("•", doc.metadata.get("source", "inconnu"))

# ============================
# Affichage de l’historique
# ============================
st.markdown("---")
st.subheader("📜 Historique des interactions")

if st.button("Voir l'historique"):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, question, answer, summary, topic FROM history ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    if rows:
        for row in rows:
            st.markdown(f"""
            🕒 **Date** : {row[0]}  
            💬 **Question** : {row[1]}  
            💡 **Réponse** : {row[2]}  
            📝 **Résumé** : {row[3]}  
            📚 **Thème** : {row[4]}  
            ---
            """)
    else:
        st.info("Aucune interaction enregistrée pour le moment.")