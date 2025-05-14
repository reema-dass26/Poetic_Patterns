import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIG ---
st.set_page_config(page_title="Poetic Patterns", layout="wide")
st.title("Poetic Patterns – v0.5")
with st.expander("ℹ️ What is *Poetic Patterns*?", expanded=True):
    st.markdown("""
    **Poetic Patterns** is an interactive tool for analyzing the emotional and stylistic layers of poetry.

    🧠 **Core Features**:
    - **Emotion detection** powered by a fine-tuned BERT model
    - **Line-by-line emotion breakdown**
    - **Emotional complexity score** to capture nuance and emotional richness
    - **Emotional landscape plot** that visually distinguishes positive vs negative lines
    - **Similarity search** to find poems with a similar emotional and stylistic vibe

    🛡️ The app also includes built-in checks for emotionally distressing content and flags them for user safety.

    ---
    📌 Paste your own poem or written thoughts below to explore their emotional fingerprint.
    """)
st.markdown("### 🪶 Analyze emotions, structure, and style in your writing.")

DATA_PATH =  "src/preprocessing/full_labeled_poetry_dataset.csv"

# MODEL_DIR = "src/Code/bert_emotion_model"

# --- Load Dataset ---
if os.path.exists(DATA_PATH):
    dataset = pd.read_csv(DATA_PATH)
else:
    dataset = None
    st.warning("Poem dataset not found!")

# --- Load Emotion Classifier ---
# @st.cache_resource
# def load_emotion_classifier():
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#     return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=3)

@st.cache_resource
def load_emotion_classifier():
    model_name = "j-hartmann/emotion-english-distilroberta-base"  # ✅ this is a valid Hugging Face model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=3)

emotion_classifier = load_emotion_classifier()

# --- Emotional Complexity Score ---
def compute_emotional_complexity(sentiments):
    emotion_labels = [label for label, _ in sentiments]
    unique_emotions = len(set(emotion_labels))
    confidence_scores = [score for _, score in sentiments]
    confidence_std = np.std(confidence_scores)
    top_count = Counter(emotion_labels).most_common(1)[0][1]
    dominance_ratio = top_count / len(emotion_labels)
    diversity_score = unique_emotions / len(emotion_labels)
    spread_score = confidence_std
    balance_score = 1 - dominance_ratio
    final_score = (0.4 * diversity_score) + (0.3 * spread_score) + (0.3 * balance_score)
    return round(final_score, 3)

# --- Similar Poem Search ---
def get_similar_poems(user_poem, dataset, top_n=3):
    vectorizer = TfidfVectorizer()
    poems = dataset["poem"].tolist()
    vectors = vectorizer.fit_transform(poems + [user_poem])
    similarities = cosine_similarity(vectors[-1], vectors[:-1])[0]
    dataset["similarity_score"] = similarities
    return dataset.sort_values(by="similarity_score", ascending=False).head(top_n)[["title", "similarity_score", "poem"]]

# --- UI Input ---
st.write("## Paste or Write Your Poem")
poem = st.text_area("Poem Text", height=300)

# --- Analysis ---
if st.button("Analyze"):
    
    if not poem.strip():
        st.warning("Please enter a poem.")
    else:
        lines = [line.strip() for line in poem.strip().split('\n') if line.strip()]
        sentiments = []
        for line in lines:
            try:
                preds = emotion_classifier(line[:512])[0]
                top_label = preds[0]["label"]
                top_score = round(preds[0]["score"], 2)
                sentiments.append((top_label, top_score))
            except:
                sentiments.append(("error", 0.0))


        # Emotional Complexity Score
        score = compute_emotional_complexity(sentiments)
        st.metric("🧠 Emotional Complexity Score", f"{score}/1.0")
        if score < 0.3:
            st.info("This poem is emotionally consistent and focused.")
        elif score < 0.6:
            st.warning("Moderate complexity — emotional variation is present.")
        else:
            st.success("High complexity — rich emotional transitions detected.")




        # Display Line-by-Line Emotion
        st.write("### Line-by-Line Emotion (BERT-Based)")
        for i, (line, (emotion, score)) in enumerate(zip(lines, sentiments)):
            st.markdown(f"**Line {i+1}:** `{line}` — Emotion: `{emotion}` ({score})")




        # Content Warning
        dangerous_emotions = {"despair", "grief", "fear", "guilt"}
        flagged = any(em in dangerous_emotions and score > 0.6 for em, score in sentiments)
        if flagged:
            st.error("⚠️ This poem may express emotional distress. If you or someone you know is struggling, please seek help.")

        # Emotion Plot
        st.write("### Emotional Landscape of the Poem")
        positive_emotions = {'joy', 'hope', 'love', 'awe', 'empowerment', 'surprise'}
        negative_emotions = {'sadness', 'grief', 'fear', 'anger', 'disgust', 'guilt', 'despair', 'yearning'}
        emotion_labels = [label for label, _ in sentiments]
        emotion_scores = [score if label in positive_emotions else -score for label, score in sentiments]
        colors = ['mediumseagreen' if label in positive_emotions else 'indianred' for label in emotion_labels]
        line_numbers = list(range(1, len(emotion_scores) + 1))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        sc = ax.scatter(line_numbers, emotion_scores, c=colors, s=120, alpha=0.9)
        for x, y, label in zip(line_numbers, emotion_scores, emotion_labels):
            ax.annotate(f"{label.title()} ({abs(y):.2f})", (x, y), textcoords="offset points", xytext=(0, 12),
                        ha='center', fontsize=9, color='black', weight='bold')
        ax.set_title("Emotional Landscape", fontsize=16, fontweight='bold')
        ax.set_xlabel("Line Number", fontsize=12)
        ax.set_ylabel("Emotion Strength (±)", fontsize=12)
        ax.set_xticks(line_numbers)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

        # Style Metrics
        st.write("### Style Metrics")
        avg_line_length = sum(len(line.split()) for line in lines) / len(lines)
        st.markdown(f"**Average line length:** {round(avg_line_length, 2)} words")

        # Overall Emotion Prediction
        st.write("### Overall Emotion Prediction")
        try:
            result = emotion_classifier(poem[:512])[0]
            st.write("#### Top Predicted Emotions with Confidence")
            for r in result:
                st.markdown(f"- **{r['label'].title()}** — `{round(r['score'] * 100, 2)}%` confidence")
            st.success(f"Primary Emotion: **{result[0]['label'].upper()}**")
        except Exception as e:
            st.error(f"Emotion prediction error: {e}")

        # Similar Poems
        if dataset is not None:
            st.write("### Most Similar Poems from Dataset")
            similar_poems = get_similar_poems(poem, dataset)
            for idx, row in similar_poems.iterrows():
                st.markdown(f"**Title:** {row['title']} — Similarity: `{round(row['similarity_score'], 2)}`")
                st.code(row['poem'])
