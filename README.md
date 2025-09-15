# Assistant Pédagogique IA (LangChain + FAISS + FLAN-T5)

##  Objectif du projet
Développer une application web permettant aux étudiants de poser des questions sur leurs cours universitaires (algorithmique, programmation, mathématiques, etc.) et d’obtenir des réponses claires, contextualisées et fiables.  

L’outil repose sur l’approche **RAG (Retrieval-Augmented Generation)** :  
- **Recherche sémantique** dans des documents indexés (FAISS + embeddings MiniLM).  
- **Génération de réponses** via un modèle de langage (FLAN-T5 local, fallback GPT-3.5).  

---

##  Architecture du système



Streamlit	Interface utilisateur (entrée de question, affichage de réponse et résumé)
LangChain	Orchestration entre les composants de recherche et de génération
HuggingFaceEmbeddings	Génération d'embeddings via all-MiniLM-L6-v2
FAISS	Index vectoriel pour la recherche sémantique rapide
FLAN-T5 (local)	Génération de réponse initiale
GPT-3.5 (OpenAI)	Fallback si la réponse locale est insuffisante
DeepTranslator	Traduction automatique FR ↔ EN
SQLite	Sauvegarde des requêtes, réponses, résumés et thématiques détectées



##  Fonctionnalités
- ** Recherche intelligente** : extraction de contexte depuis les documents vectorisés.
- ** Génération de réponse** : par FLAN-T5 (ou GPT-3.5 en fallback).
- ** Support multilingue** : traduction automatique si question en français.
- ** Résumé synthétique** : résumé automatique des réponses.
- ** Classification par thème** : détection (programmation, maths, algorithmique, etc.).
- ** Historique SQLite** : stockage avec horodatage, thème, résumé et question.

##  Démonstration
### Exemple 1 : Question simple
- **Q** : C’est quoi un algorithme ?
- **Réponse** : Un algorithme est une méthode pour résoudre un problème.
- **Résumé** : A method for resolving a problem.
- **Thème détecté** : Algorithmique
- **Modèle utilisé** : FLAN-T5

### Exemple 2 : Question en programmation
- **Q** : C'est quoi Python ?
- **Réponse** : Python est un langage portable, extensible et orienté objet.
- **Résumé** : Python is a portable, extensible, object-oriented language.
- **Thème détecté** : Programmation

##  Base de données SQLite (history.db)
Chaque interaction est sauvegardée :

| Champ     | Type  | Description                       |
|-----------|-------|-----------------------------------|
| timestamp | TEXT  | Date et heure de la requête      |
| question  | TEXT  | Question posée                   |
| answer    | TEXT  | Réponse générée                  |
| summary   | TEXT  | Résumé de la réponse             |
| topic     | TEXT  | Thématique détectée              |

##  Installation et lancement
1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/<user>/<repo>.git
   cd <repo>
   ```
2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```
3. **Lancer l’application**
   ```bash
   streamlit run app.py
   ```

##  Limites actuelles
- Réponses de FLAN-T5 parfois trop courtes.
- Traduction automatique encore basique.
- Fallback GPT-3.5 nécessite une clé API valide.
- Détection de thématiques basée sur mots-clés simples.

##  Perspectives
- Fine-tuning du modèle local (ex: Mistral, Phi-2).
- Classification thématique supervisée.
- Dashboard analytique (Streamlit ou Grafana).
- Authentification multi-utilisateurs.
- Export PDF ou envoi par email des réponses.

##  Auteurs
Projet réalisé par Keryl Jalbred – Étudiante Data Engineer/IA.
