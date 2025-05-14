
# 🪶 Poetic Patterns

**Poetic Patterns** is an interactive Streamlit web app that analyzes poems using machine learning and NLP to uncover their emotional landscape, structural style, and artistic depth.

---

## 🔍 Features

- ✨ **Emotion Classification** using a fine-tuned BERT-based model
- 📈 **Emotional Landscape Plot** – positive vs negative emotion arcs
- 🧠 **Emotional Complexity Score** – diversity, dominance, and variation
- 🧮 **Style Metrics** – line length analysis
- 🧑‍🤝‍🧑 **Similar Poem Search** – cosine similarity from labeled corpus
- ⚠️ **Mental Health Flagging** – detects potentially distressing content

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/poetic-patterns.git
cd poetic-patterns
pip install -r requirements.txt

streamlit run app.py


├── app.py                      # Streamlit app
├── src/
│   ├── Code/
│   │   └── bert_emotion_model/  # Saved BERT model (downloaded)
│   └── preprocessing/
│       └── data_cleaned/        # Cleaned CSV poetry dataset
├── requirements.txt
└── README.md
