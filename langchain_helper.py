import os
import pickle
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import re
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsRAG:
    def __init__(self):
        """Initialize the News RAG system"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"}
        )

        self.vector_store = None
        self.load_vector_store()
        self.articles_data = self.load_articles_data()
        self.init_gemini()

        self.query_templates = {
            'summary': "Provide a comprehensive summary of the following news articles:\n\n{context}",
            'headlines': "Extract and list the main headlines from the following news content:\n\n{context}",
            'specific': "Based on the following news articles, answer this question: {question}\n\nNews content:\n{context}",
            'today': "From the following recent news articles, provide today's key news highlights:\n\n{context}"
        }

    def init_gemini(self):
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                logger.info("Gemini API initialized successfully")
            else:
                logger.warning("Gemini API key not found in environment variables")
                self.model = None
        except Exception as e:
            logger.error(f"Error initializing Gemini: {str(e)}")
            self.model = None

    def load_vector_store(self):
        try:
            if os.path.exists("faiss_index"):
                self.vector_store = FAISS.load_local(
                    "faiss_index",
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector store loaded successfully")
            else:
                logger.error("Vector store not found at 'faiss_index' directory")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")

    def load_articles_data(self):
        try:
            if os.path.exists('articles_data.pickle'):
                with open('articles_data.pickle', 'rb') as f:
                    articles = pickle.load(f)
                logger.info(f"Loaded {len(articles)} articles")
                return articles
            else:
                logger.warning("Articles data not found at 'articles_data.pickle'")
                return []
        except Exception as e:
            logger.error(f"Error loading articles data: {str(e)}")
            return []

    def classify_query(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ['headline', 'headlines', 'top news', 'main news']):
            return 'headlines'
        elif any(word in query_lower for word in ['today', "today's", 'current', 'latest', 'recent']):
            return 'today'
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview', 'brief']):
            return 'summary'
        else:
            return 'specific'

    def enhance_query(self, query: str) -> str:
        enhanced_query = query
        if any(word in query.lower() for word in ['today', 'current', 'latest', 'recent']):
            enhanced_query += f" {datetime.now().strftime('%Y-%m-%d')}"
        if not any(word in query.lower() for word in ['news', 'article', 'report']):
            enhanced_query += " news article"
        return enhanced_query

    def retrieve_relevant_documents(self, query: str, k: int = 10) -> List[Document]:
        if not self.vector_store:
            logger.error("Vector store not loaded!")
            return []

        try:
            enhanced_query = self.enhance_query(query)
            docs = self.vector_store.similarity_search(enhanced_query, k=k)

            if any(word in query.lower() for word in ['today', 'current', 'latest']):
                docs = self.filter_recent_docs(docs)

            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def filter_recent_docs(self, docs: List[Document], days_back: int = 7) -> List[Document]:
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_docs = []

        for doc in docs:
            try:
                published = doc.metadata.get('published', '')
                if published:
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%a, %d %b %Y %H:%M:%S %Z']:
                        try:
                            doc_date = datetime.strptime(published, fmt)
                            if doc_date >= cutoff_date:
                                recent_docs.append(doc)
                            break
                        except ValueError:
                            continue
                else:
                    recent_docs.append(doc)
            except Exception:
                recent_docs.append(doc)

        return recent_docs[:len(docs)//2] if recent_docs else docs

    def prepare_context(self, docs: List[Document], query_type: str) -> str:
        if not docs:
            return "No relevant articles found."

        context_parts = []
        seen_titles = set()

        for doc in docs:
            title = doc.metadata.get('title', 'Unknown Title')
            if title in seen_titles:
                continue
            seen_titles.add(title)

            source = doc.metadata.get('source', 'Unknown Source')
            url = doc.metadata.get('url', '')
            published = doc.metadata.get('published', '')

            if query_type == 'headlines':
                context_parts.append(f"\u2022 {title} ({source})")
            else:
                article_text = f"**{title}**\n"
                article_text += f"Source: {source}\n"
                if published:
                    article_text += f"Published: {published}\n"
                article_text += f"Content: {doc.page_content}\n"
                if url:
                    article_text += f"URL: {url}\n"
                article_text += "---\n"
                context_parts.append(article_text)

        return "\n".join(context_parts)

    def generate_response(self, query: str, context: str, query_type: str) -> str:
        if not self.model:
            return "Gemini API not available. Please check your API key."

        try:
            template = self.query_templates.get(query_type, self.query_templates['specific'])
            prompt = template.format(question=query, context=context) if query_type == 'specific' else template.format(context=context)
            instructions = """
            Please provide a comprehensive and well-structured response based on the news articles provided. Don't add anything from your side. 
            If you're asked for headlines, provide a clear list.
            If you're asked for a summary, provide key points and important details.
            If you're asked a specific question, answer it directly using the information from the articles.
            Always mention the sources when relevant.
            No Preamble. Direct answer.
            """
            full_prompt = instructions + "\n\n" + prompt
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def query(self, query: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing query: {query}")
            query_type = self.classify_query(query)
            logger.info(f"Query type: {query_type}")
            docs = self.retrieve_relevant_documents(query, k=15)

            if not docs:
                return {
                    'answer': "I couldn't find any relevant news articles for your query. Please try a different question or check if the vector store has been built.",
                    'sources': [],
                    'query_type': query_type
                }

            context = self.prepare_context(docs, query_type)
            answer = self.generate_response(query, context, query_type)

            sources = []
            seen_urls = set()
            for doc in docs[:5]:
                url = doc.metadata.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        'title': doc.metadata.get('title', 'Unknown Title'),
                        'source': doc.metadata.get('source', 'Unknown Source'),
                        'url': url,
                        'published': doc.metadata.get('published', ''),
                        'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                    })

            return {
                'answer': answer,
                'sources': sources,
                'query_type': query_type,
                'context_length': len(context)
            }

        except Exception as e:
            logger.error(f"Error in query method: {str(e)}")
            return {
                'answer': f"An error occurred while processing your query: {str(e)}",
                'sources': [],
                'query_type': 'error'
            }

def news_ai_agent(query: str) -> Dict[str, Any]:
    rag = NewsRAG()
    result = rag.query(query)
    formatted_sources = []
    for source in result.get('sources', []):
        formatted_sources.append({
            'heading': source.get('title', 'Unknown Title'),
            'source': source.get('source', 'Unknown Source'),
            'url': source.get('url', ''),
            'published': source.get('published', ''),
            'content': source.get('content', '')
        })

    return {
        'answer': result.get('answer', 'No answer generated'),
        'sources': formatted_sources
    }
