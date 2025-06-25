# 📰 News AI Assistant

An advanced RAG-powered news assistant that scrapes real-time articles from major news websites and allows users to interact via a chatbot interface to ask questions, get answers, and explore the latest headlines — all enhanced by cutting-edge query generation and retrieval techniques.

🌐 **Live App**: [yashodeep-news-ai.streamlit.app](https://yashodeep-news-ai.streamlit.app)

---

## 🚀 Features

- 🌍 Real-time news scraping from various top websites and saved to github
- 🧠 Advanced Retrieval-Augmented Generation (RAG)
- 🧩 Query expansion using **HyDE (Hypothetical Document Embeddings)**
- ⚡ Fast semantic search via **FAISS**
- 🤖 Uses both Hugging Face models and **Gemini API**
- 💬 Simple, responsive chatbot UI built and deployed with Streamlit

---

## 🔧 Tech Stack

- **Python**
- **BeautifulSoup** – News scraping
- **FAISS** – Vector search indexing
- **LangChain** – RAG pipeline and query management
- **HuggingFace Transformers** – Embedding 
- **Gemini API** – LLM for HyDE query generation and answer generation
- **Streamlit** – Frontend UI

---

## ⚙️ How It Works

1. **News Scraper** gathers articles using `requests` and `BeautifulSoup`.
2. **Embeddings** are created using HuggingFace models.
3. **FAISS Index** stores vectors locally for quick retrieval.
4. **HyDE** is used to generate hypothetical Document to expand search effectiveness and get exact chunks.
5. **RAG Pipeline** retrieves and feeds relevant data into LLM.
6. **User Interface** displays responses and maintains context chat.
7. **Data** Saves data into file which is updated in the github for everyone.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/news-ai-assistant.git
cd news-ai-assistant
pip install -r requirements.txt
```

.env file setup - 

```bash
HUGGINGFACEHUB_API_TOKEN="YOUR_API_KEY"
GEMINI_API_KEY = "YOUR_API_KEY"
GITHUB_URL="YOUR_GITHUB_REPO_URL"
GITHUB_TOKEN="YOUR_GITHUB_API_KEY"
