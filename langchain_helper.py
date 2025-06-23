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
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsRAG:
    def __init__(self):
        """Initialize the News RAG system"""
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        if hf_token: 
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},
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
            api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                logger.info("Gemini API initialized successfully")
            else:
                logger.warning("Gemini API key not found in environment variables or Streamlit secrets")
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
        """Enhanced query expansion with better category and time handling"""
        enhanced_query = query
        
        # Add date context for time-sensitive queries
        if any(word in query.lower() for word in ['today', 'current', 'latest', 'recent']):
            enhanced_query += f" {datetime.now().strftime('%Y-%m-%d')}"
        
        # Add context for news-related queries
        if not any(word in query.lower() for word in ['news', 'article', 'report']):
            enhanced_query += " news article"
        
        return enhanced_query

    def retrieve_relevant_documents(self, query: str, k: int = 15) -> List[Document]:
        """Enhanced document retrieval with better filtering"""
        if not self.vector_store:
            logger.error("Vector store not loaded!")
            return []

        try:
            enhanced_query = self.enhance_query(query)
            docs = self.vector_store.similarity_search(enhanced_query, k=k*2)  # Get more docs initially

            # Filter by time constraints
            if any(word in query.lower() for word in ['today', 'current', 'latest']):
                docs = self.filter_recent_docs(docs, days_back=1)  # More strict for "today"
            elif 'recent' in query.lower():
                docs = self.filter_recent_docs(docs, days_back=7)

            # Return top k documents after filtering
            filtered_docs = docs[:k]
            logger.info(f"Retrieved {len(filtered_docs)} relevant documents after filtering")
            return filtered_docs

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

        return recent_docs if recent_docs else docs[:len(docs)//2]

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
            
            # Enhanced instructions for better response generation
            instructions = f"""
            Please provide a comprehensive and well-structured response based ONLY on the news articles provided. 
            IMPORTANT: Only use information that is directly present in the provided articles.
            
            Query: {query}
            
            Guidelines:
            - If you're asked for headlines, provide a clear list.
            - If you're asked for a summary, provide key points and important details.
            - If you're asked a specific question, answer it directly using ONLY the information from the articles.
            - Always mention the sources when relevant.
            - If the articles don't contain information to answer the specific query, say so clearly.
            - No preamble. Direct answer.
            - Focus on the specific topic requested (e.g., if sports is asked, focus only on sports content).
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
                    'answer': "No relevant articles found for your query. Please try rephrasing your question or check for different keywords.",
                    'sources': [],
                    'query_type': query_type,
                    'is_from_sources': False
                }

            context = self.prepare_context(docs, query_type)
            answer = self.generate_response(query, context, query_type)

            # Prepare sources
            sources = []
            seen_urls = set()
            for doc in docs[:8]:
                url = doc.metadata.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({
                        'title': doc.metadata.get('title', 'Unknown Title'),
                        'source': doc.metadata.get('source', 'Unknown Source'),
                        'url': url,
                        'published': doc.metadata.get('published', ''),
                        'content': doc.page_content
                    })

            return {
                'answer': answer,
                'sources': sources,
                'query_type': query_type,
                'context_length': len(context),
                'is_from_sources': True
            }

        except Exception as e:
            logger.error(f"Error in query method: {str(e)}")
            return {
                'answer': f"An error occurred while processing your query: {str(e)}",
                'sources': [],
                'query_type': 'error',
                'is_from_sources': False
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
        'sources': formatted_sources,
        'is_from_sources': result.get('is_from_sources', False)
    }
