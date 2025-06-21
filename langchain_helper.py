import os
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsRAG:
    def __init__(self):
        """Initialize the News RAG system"""
        # Initialize Gemini
        self.init_gemini()
        
        # Initialize embeddings (cached)
        self.init_embeddings()
        
        # Load data
        self.vector_store = None
        self.articles_data = []
        self.load_data()
        
        # Query categories for filtering
        self.categories = {
            'sports': ['sports', 'football', 'cricket', 'match', 'game', 'tournament'],
            'politics': ['politics', 'election', 'government', 'minister', 'president'],
            'business': ['business', 'economy', 'stock', 'market', 'finance', 'company'],
            'technology': ['technology', 'tech', 'ai', 'software', 'startup'],
            'health': ['health', 'medical', 'medicine', 'hospital', 'covid'],
            'entertainment': ['entertainment', 'movie', 'celebrity', 'music', 'show']
        }

    @st.cache_resource
    def init_embeddings(_self):
        """Initialize embeddings with caching"""
        try:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller, faster model
                model_kwargs={"device": "cpu"}
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            return None

    def init_gemini(self):
        """Initialize Gemini API"""
        try:
            api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Gemini initialized")
            else:
                self.model = None
                logger.warning("Gemini API key not found")
        except Exception as e:
            logger.error(f"Gemini init error: {e}")
            self.model = None

    def load_data(self):
        """Load vector store and articles data"""
        try:
            # Load vector store
            if os.path.exists("faiss_index"):
                self.embeddings = self.init_embeddings()
                if self.embeddings:
                    self.vector_store = FAISS.load_local(
                        "faiss_index", 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info("Vector store loaded")
            
            # Load articles data
            if os.path.exists('articles_data.pickle'):
                with open('articles_data.pickle', 'rb') as f:
                    self.articles_data = pickle.load(f)
                logger.info(f"Loaded {len(self.articles_data)} articles")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def get_query_category(self, query: str) -> str:
        """Get query category"""
        query_lower = query.lower()
        for category, keywords in self.categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return 'general'

    def is_recent_query(self, query: str) -> bool:
        """Check if query asks for recent/today's news"""
        recent_words = ['today', 'latest', 'recent', 'current', 'breaking']
        return any(word in query.lower() for word in recent_words)

    def retrieve_documents(self, query: str, k: int = 10) -> List[Document]:
        """Retrieve relevant documents"""
        if not self.vector_store:
            return []
        
        try:
            # Get documents
            docs = self.vector_store.similarity_search(query, k=k*2)
            
            # Filter by recency if needed
            if self.is_recent_query(query):
                docs = self.filter_recent_docs(docs)
            
            # Filter by category
            category = self.get_query_category(query)
            if category != 'general':
                docs = self.filter_by_category(docs, category)
            
            return docs[:k]
        except Exception as e:
            logger.error(f"Document retrieval error: {e}")
            return []

    def filter_recent_docs(self, docs: List[Document], days: int = 3) -> List[Document]:
        """Filter documents by recency"""
        cutoff = datetime.now() - timedelta(days=days)
        recent_docs = []
        
        for doc in docs:
            try:
                published = doc.metadata.get('published', '')
                if published:
                    # Simple date parsing
                    if any(format_str in published for format_str in ['2024', '2025']):
                        doc_date = datetime.strptime(published.split()[0], '%Y-%m-%d')
                        if doc_date >= cutoff:
                            recent_docs.append(doc)
                else:
                    recent_docs.append(doc)  # Include if no date
            except:
                recent_docs.append(doc)  # Include if parsing fails
        
        return recent_docs if recent_docs else docs[:len(docs)//2]

    def filter_by_category(self, docs: List[Document], category: str) -> List[Document]:
        """Filter documents by category"""
        if category not in self.categories:
            return docs
        
        keywords = self.categories[category]
        filtered = []
        
        for doc in docs:
            text = (doc.page_content + ' ' + doc.metadata.get('title', '')).lower()
            if any(keyword in text for keyword in keywords):
                filtered.append(doc)
        
        return filtered if filtered else docs

    def prepare_context(self, docs: List[Document]) -> str:
        """Prepare context from documents"""
        if not docs:
            return "No articles found."
        
        context_parts = []
        seen_titles = set()
        
        for doc in docs[:8]:  # Limit to 8 articles
            title = doc.metadata.get('title', 'Unknown')
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content[:500]  # Limit content length
            
            context_parts.append(f"Title: {title}\nSource: {source}\nContent: {content}\n---")
        
        return "\n".join(context_parts)

    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Gemini"""
        if not self.model:
            return "⚠️ AI model not available. Please check configuration."
        
        try:
            prompt = f"""Based on the following news articles, answer the question: {query}

Guidelines:
- Use only information from the provided articles
- Be concise and factual
- Mention sources when relevant
- If information is not available, say so clearly

Articles:
{context}

Answer:"""
            
            response = self.model.generate_content(prompt)
            return response.text if response.text else "No response generated."
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return f"⚠️ Error generating answer. Please try again."

    def prepare_sources(self, docs: List[Document]) -> List[Dict]:
        """Prepare source information"""
        sources = []
        seen_urls = set()
        
        for doc in docs[:5]:  # Limit to 5 sources
            url = doc.metadata.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    'title': doc.metadata.get('title', 'Unknown'),
                    'source': doc.metadata.get('source', 'Unknown'),
                    'url': url,
                    'published': doc.metadata.get('published', '')
                })
        
        return sources

    def query(self, query: str) -> Dict[str, Any]:
        """Main query method"""
        try:
            logger.info(f"Query: {query}")
            
            # Retrieve documents
            docs = self.retrieve_documents(query)
            
            if not docs:
                return {
                    'answer': "⚠️ No relevant articles found. Please try a different search term or check back later.",
                    'sources': [],
                    'success': False
                }
            
            # Generate answer
            context = self.prepare_context(docs)
            answer = self.generate_answer(query, context)
            sources = self.prepare_sources(docs)
            
            return {
                'answer': answer,
                'sources': sources,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                'answer': f"⚠️ Error processing query: {str(e)}",
                'sources': [],
                'success': False
            }

def news_ai_agent(query: str) -> Dict[str, Any]:
    """Main function to get news AI response"""
    try:
        rag = NewsRAG()
        result = rag.query(query)
        
        # Format for compatibility
        formatted_sources = []
        for source in result.get('sources', []):
            formatted_sources.append({
                'heading': source.get('title', ''),
                'source': source.get('source', ''),
                'url': source.get('url', ''),
                'published': source.get('published', '')
            })
        
        return {
            'answer': result.get('answer', 'No answer available'),
            'sources': formatted_sources
        }
        
    except Exception as e:
        logger.error(f"Main function error: {e}")
        return {
            'answer': f"⚠️ Service temporarily unavailable: {str(e)}",
            'sources': []
        }
