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
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerificationAgent:
    def __init__(self):
        """Initialize the verification agent with a fast, efficient LLM"""
        try:
            # Using Microsoft's DialoGPT-medium for fast response generation
            # Alternative: Use "microsoft/DialoGPT-small" for even faster responses
            model_name = "microsoft/DialoGPT-medium"
            
            # Check if CUDA is available for faster processing
            device = 0 if torch.cuda.is_available() else -1
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Alternative: Use a text-generation pipeline for simpler implementation
            # self.generator = pipeline(
            #     "text-generation",
            #     model=model_name,
            #     tokenizer=self.tokenizer,
            #     device=device,
            #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            #     max_length=512,
            #     do_sample=True,
            #     temperature=0.7
            # )
            
            logger.info(f"Verification agent initialized with {model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing verification agent: {str(e)}")
            # Fallback to a simpler approach
            try:
                self.generator = pipeline(
                    "text-generation",
                    model="gpt2",
                    device=device,
                    max_length=256
                )
                logger.info("Fallback to GPT-2 for verification agent")
            except Exception as e2:
                logger.error(f"Fallback initialization failed: {str(e2)}")
                self.generator = None
                self.model = None
                self.tokenizer = None

    def check_relevance_and_quality(self, query: str, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """Check if the answer is relevant to the query and has sufficient quality"""
        try:
            # Simple relevance check based on keyword overlap
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            
            # Remove common stop words for better relevance scoring
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            query_words -= stop_words
            answer_words -= stop_words
            
            # Calculate relevance score
            if len(query_words) > 0:
                relevance_score = len(query_words.intersection(answer_words)) / len(query_words)
            else:
                relevance_score = 0.0
            
            # Check answer quality (length, structure, etc.)
            quality_score = 0.0
            if len(answer) > 50:  # Minimum length check
                quality_score += 0.3
            if len(answer.split('.')) > 1:  # Multiple sentences
                quality_score += 0.3
            if any(source.get('title', '') in answer for source in sources):  # References sources
                quality_score += 0.4
            
            # Determine if answer is sufficient
            is_relevant = relevance_score >= 0.3  # At least 30% keyword overlap
            is_quality = quality_score >= 0.5  # At least 50% quality score
            has_sources = len(sources) > 0
            
            return {
                'is_sufficient': is_relevant and is_quality and has_sources,
                'relevance_score': relevance_score,
                'quality_score': quality_score,
                'has_sources': has_sources,
                'needs_fallback': not (is_relevant and is_quality and has_sources)
            }
            
        except Exception as e:
            logger.error(f"Error in relevance check: {str(e)}")
            return {
                'is_sufficient': False,
                'relevance_score': 0.0,
                'quality_score': 0.0,
                'has_sources': False,
                'needs_fallback': True
            }

    def beautify_answer(self, answer: str) -> str:
        """Beautify the answer without changing the information"""
        try:
            # Simple beautification: improve formatting and structure
            beautified = answer.strip()
            
            # Add proper spacing after periods if missing
            beautified = re.sub(r'\.([A-Z])', r'. \1', beautified)
            
            # Ensure proper paragraph breaks
            sentences = beautified.split('. ')
            if len(sentences) > 3:
                # Group sentences into paragraphs (every 2-3 sentences)
                paragraphs = []
                current_paragraph = []
                
                for i, sentence in enumerate(sentences):
                    current_paragraph.append(sentence)
                    if (i + 1) % 3 == 0 or i == len(sentences) - 1:
                        paragraphs.append('. '.join(current_paragraph))
                        current_paragraph = []
                
                beautified = '\n\n'.join(paragraphs)
            
            # Add bullet points for lists if detected
            if '\n•' in beautified or '\n-' in beautified:
                lines = beautified.split('\n')
                formatted_lines = []
                for line in lines:
                    if line.strip().startswith('•') or line.strip().startswith('-'):
                        formatted_lines.append(f"  {line.strip()}")
                    else:
                        formatted_lines.append(line)
                beautified = '\n'.join(formatted_lines)
            
            return beautified
            
        except Exception as e:
            logger.error(f"Error beautifying answer: {str(e)}")
            return answer

    def generate_fallback_answer(self, query: str) -> str:
        """Generate a fallback answer when sources are insufficient"""
        try:
            if not self.model or not self.tokenizer:
                return f"I don't have sufficient news sources to answer your query about '{query}'. However, based on my general knowledge, I'd recommend checking recent news outlets for the most current information on this topic."
            
            # Create a prompt for generating a helpful response
            prompt = f"User asked: {query}\n\nSince I don't have recent news sources for this query, here's what I can share based on general knowledge:"
            
            # Tokenize and generate
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=200, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 150,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            fallback_answer = generated_text[len(prompt):].strip()
            
            # Add disclaimer
            disclaimer = "⚠️ **Note**: This information is not from recent news sources but based on my general knowledge. For the most current information, please check latest news outlets."
            
            return f"{disclaimer}\n\n{fallback_answer}"
            
        except Exception as e:
            logger.error(f"Error generating fallback answer: {str(e)}")
            return f"I don't have sufficient recent news sources to answer your query about '{query}'. Please try a different search term or check the latest news directly from reliable news sources."

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
        
        # Initialize verification agent
        self.verification_agent = VerificationAgent()

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
                # No documents found, use verification agent for fallback
                fallback_answer = self.verification_agent.generate_fallback_answer(query)
                return {
                    'answer': fallback_answer,
                    'sources': [],
                    'query_type': query_type,
                    'is_from_sources': False,
                    'verification_status': 'fallback_generated'
                }

            context = self.prepare_context(docs, query_type)
            initial_answer = self.generate_response(query, context, query_type)

            # Prepare sources
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

            # Verify the answer quality and relevance
            verification_result = self.verification_agent.check_relevance_and_quality(query, initial_answer, sources)
            
            if verification_result['is_sufficient']:
                # Answer is good, beautify it
                final_answer = self.verification_agent.beautify_answer(initial_answer)
                verification_status = 'verified_and_beautified'
                is_from_sources = True
            else:
                # Answer is not sufficient, generate fallback
                fallback_answer = self.verification_agent.generate_fallback_answer(query)
                final_answer = fallback_answer
                verification_status = 'insufficient_fallback_generated'
                is_from_sources = False
                sources = []  # Clear sources since we're not using them

            return {
                'answer': final_answer,
                'sources': sources,
                'query_type': query_type,
                'context_length': len(context),
                'is_from_sources': is_from_sources,
                'verification_status': verification_status,
                'verification_details': verification_result
            }

        except Exception as e:
            logger.error(f"Error in query method: {str(e)}")
            # Generate fallback for errors
            fallback_answer = self.verification_agent.generate_fallback_answer(query)
            return {
                'answer': fallback_answer,
                'sources': [],
                'query_type': 'error',
                'is_from_sources': False,
                'verification_status': 'error_fallback_generated'
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
        'is_from_sources': result.get('is_from_sources', False),
        'verification_status': result.get('verification_status', 'unknown')
    }
