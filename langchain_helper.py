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
            'specific': "Based on the following news articles, answer this question: {question}\n\nNews content:\n{context}"
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



    def generate_hypothetical_document(self, query: str) -> Dict[str, Any]:
        """
        Generate a hypothetical document using HyDE approach.
        This creates an ideal answer that helps retrieve better documents.
        Returns a dictionary with answer and boolean indicating if it's news-related.
        """
        if not self.model:
            logger.warning("Gemini API not available for HyDE generation")
            return {"answer": query, "is_news_related": True}
        
        try:
            hyde_prompt = f"""
            You are a news expert. Given the following question or topic, first determine if this is a news-related query.
            If the query is not related to news then generate the answer as per your knowledge. You are also a human.
            
            
            If it IS news-related: write a detailed, informative news article excerpt that would perfectly answer this question. 
            Write it as if it's from a real news article with specific details, facts, and context.
            
            If it is NOT news-related: provide a direct answer based on your general knowledge.
            
            Question/Topic: {query}
            
            Strictly Respond ONLY with valid python dictionary format in exactly this format (no extra text, no markdown, no explanations):
            {{
                "answer": "your response here (either hypothetical news article excerpt for news queries or direct answer for non-news queries)",
                "is_news_related": true/false
            }}
            
            For news-related queries: Write a comprehensive news article excerpt (2-3 paragraphs) that would ideally contain the answer to this question.
            For non-news queries: Provide a direct, helpful answer based on your knowledge.
            """
            
            response = self.model.generate_content(hyde_prompt)
            response_text = response.text.strip()
            print(response_text)
            
            # Try to parse JSON response
            try:
                result = json.loads(response_text)
                logger.info(f"Generated response for query: {query[:50]}... (News-related: {result.get('is_news_related', True)})")
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse JSON response, treating as news-related")
                return {"answer": response_text, "is_news_related": True}
            
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {str(e)}")
            return {"answer": query, "is_news_related": True}

    def retrieve_relevant_documents(self, query: str, k: int = 15) -> List[Document]:
        """Enhanced document retrieval using HyDE approach"""
        if not self.vector_store:
            logger.error("Vector store not loaded!")
            return []

        try:
            # Generate hypothetical document using HyDE
            hyde_result = self.generate_hypothetical_document(query)
            hypothetical_doc = hyde_result["answer"]
            
            # Use the hypothetical document for similarity search
            docs = self.vector_store.similarity_search(hypothetical_doc, k=k)

            logger.info(f"Retrieved {len(docs)} relevant documents using HyDE")
            return docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []



    def prepare_context(self, docs: List[Document]) -> str:
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

    def generate_response(self, query: str, context: str) -> str:
        if not self.model:
            return "Gemini API not available. Please check your API key."

        try:
            prompt = f"Based on the following news articles, answer this question: {query}\n\nNews content:\n{context}"
            
            # Enhanced instructions for better response generation
            instructions = f"""
            Please provide a comprehensive and well-structured response based ONLY on the news articles provided. 
            IMPORTANT: Only use information that is directly present in the provided articles.
            
            Query: {query}
            
            Guidelines:
            - Answer the question directly using ONLY the information from the articles.
            - Always mention the sources when relevant.
            - If the articles don't contain information to answer the specific query, say so clearly.
            - No preamble. Direct answer. Reduce redundancy in your answer. Don''t write that I provided you articles.
            - Highlight important points and keywords in your answer. Make your answer presentable.
            - Focus on the specific topic requested.
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
            
            # First check if the query is news-related
            hyde_result = self.generate_hypothetical_document(query)
            
            # If not news-related, return the direct answer
            if not hyde_result.get("is_news_related", True):
                return {
                    'answer': hyde_result["answer"],
                    'sources': [],
                    'is_from_sources': False
                }
            
            # Continue with normal flow for news-related queries
            docs = self.retrieve_relevant_documents(query, k=15)

            if not docs:
                return {
                    'answer': "No relevant articles found for your query. Please try rephrasing your question or check for different keywords.",
                    'sources': [],
                    'is_from_sources': False
                }

            context = self.prepare_context(docs)
            answer = self.generate_response(query, context)

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
                'context_length': len(context),
                'is_from_sources': True
            }

        except Exception as e:
            logger.error(f"Error in query method: {str(e)}")
            return {
                'answer': f"An error occurred while processing your query: {str(e)}",
                'sources': [],
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
