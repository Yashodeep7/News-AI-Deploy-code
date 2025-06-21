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
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VerificationAgent:
    def __init__(self):
        """Initialize the verification agent with a fast, efficient LLM"""
        try:
            # Get HuggingFace token from environment or Streamlit secrets
            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
            if hf_token:
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
            
            # Using Microsoft's DialoGPT-medium for fast response generation
            model_name = "microsoft/DialoGPT-medium"
            
            # Use CPU for Streamlit deployment compatibility
            device = -1  # Force CPU usage for Streamlit
            
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                device=device,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256  # Set pad token for DialoGPT
            )
            
            logger.info(f"Verification agent initialized with {model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing verification agent: {str(e)}")
            # Fallback to GPT-2 for better Streamlit compatibility
            try:
                self.generator = pipeline(
                    "text-generation",
                    model="gpt2",
                    device=-1,  # Force CPU
                    max_length=256,
                    pad_token_id=50256
                )
                logger.info("Fallback to GPT-2 for verification agent")
            except Exception as e2:
                logger.error(f"Fallback initialization failed: {str(e2)}")
                self.generator = None

    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract specific intent and constraints from the query"""
        query_lower = query.lower()
        
        # Extract category/topic constraints
        categories = {
            'sports': ['sports', 'football', 'cricket', 'basketball', 'tennis', 'soccer', 'baseball', 'hockey', 'athletics', 'olympics', 'fifa', 'match', 'game', 'tournament', 'championship'],
            'politics': ['politics', 'election', 'government', 'minister', 'president', 'congress', 'parliament', 'policy', 'vote', 'campaign'],
            'business': ['business', 'economy', 'stock', 'market', 'finance', 'company', 'corporate', 'revenue', 'profit', 'investment', 'banking'],
            'technology': ['technology', 'tech', 'ai', 'artificial intelligence', 'software', 'hardware', 'startup', 'internet', 'cyber', 'digital'],
            'health': ['health', 'medical', 'medicine', 'hospital', 'doctor', 'disease', 'covid', 'vaccine', 'treatment', 'healthcare'],
            'entertainment': ['entertainment', 'movie', 'film', 'celebrity', 'music', 'show', 'actor', 'actress', 'hollywood', 'bollywood']
        }
        
        detected_categories = []
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_categories.append(category)
        
        # Extract time constraints
        time_constraints = []
        if any(word in query_lower for word in ['today', "today's", 'current']):
            time_constraints.append('today')
        elif any(word in query_lower for word in ['yesterday', "yesterday's"]):
            time_constraints.append('yesterday')
        elif any(word in query_lower for word in ['this week', 'weekly']):
            time_constraints.append('this_week')
        elif any(word in query_lower for word in ['latest', 'recent']):
            time_constraints.append('recent')
        
        # Extract location constraints
        locations = []
        location_keywords = ['india', 'indian', 'usa', 'america', 'american', 'uk', 'britain', 'british', 'china', 'chinese', 'japan', 'japanese', 'europe', 'european', 'local', 'international', 'global', 'world']
        for location in location_keywords:
            if location in query_lower:
                locations.append(location)
        
        return {
            'categories': detected_categories,
            'time_constraints': time_constraints,
            'locations': locations,
            'original_query': query
        }

    def validate_content_relevance(self, query_intent: Dict[str, Any], answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """Validate if the answer content matches the query intent"""
        answer_lower = answer.lower()
        
        # Check category relevance
        category_match = False
        if query_intent['categories']:
            categories = {
                'sports': ['sports', 'football', 'cricket', 'basketball', 'tennis', 'soccer', 'baseball', 'hockey', 'athletics', 'olympics', 'fifa', 'match', 'game', 'tournament', 'championship', 'player', 'team', 'score', 'win', 'loss'],
                'politics': ['politics', 'election', 'government', 'minister', 'president', 'congress', 'parliament', 'policy', 'vote', 'campaign', 'political', 'politician'],
                'business': ['business', 'economy', 'stock', 'market', 'finance', 'company', 'corporate', 'revenue', 'profit', 'investment', 'banking', 'economic', 'financial'],
                'technology': ['technology', 'tech', 'ai', 'artificial intelligence', 'software', 'hardware', 'startup', 'internet', 'cyber', 'digital', 'computer', 'app'],
                'health': ['health', 'medical', 'medicine', 'hospital', 'doctor', 'disease', 'covid', 'vaccine', 'treatment', 'healthcare', 'patient'],
                'entertainment': ['entertainment', 'movie', 'film', 'celebrity', 'music', 'show', 'actor', 'actress', 'hollywood', 'bollywood', 'cinema']
            }
            
            for required_category in query_intent['categories']:
                if required_category in categories:
                    category_keywords = categories[required_category]
                    # Check if answer contains sufficient category-specific keywords
                    found_keywords = [kw for kw in category_keywords if kw in answer_lower]
                    if len(found_keywords) >= 2:  # At least 2 relevant keywords
                        category_match = True
                        break
        else:
            category_match = True  # No specific category required
        
        # Check time relevance for sources
        time_relevance = True
        if 'today' in query_intent['time_constraints']:
            today = datetime.now().date()
            recent_sources = 0
            for source in sources:
                published = source.get('published', '')
                if published:
                    try:
                        # Try different date formats
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%a, %d %b %Y %H:%M:%S %Z']:
                            try:
                                source_date = datetime.strptime(published.strip(), fmt).date()
                                if source_date >= today - timedelta(days=1):  # Today or yesterday
                                    recent_sources += 1
                                break
                            except ValueError:
                                continue
                    except:
                        pass
            
            # At least 30% of sources should be recent for "today's" queries
            if sources and recent_sources / len(sources) < 0.3:
                time_relevance = False
        
        # Check if answer is too generic
        generic_phrases = [
            'based on general knowledge',
            'i don\'t have sufficient',
            'please check latest news',
            'i cannot provide',
            'not available in my knowledge',
            'here\'s what i can share',
            'based on my training data'
        ]
        is_generic = any(phrase in answer_lower for phrase in generic_phrases)
        
        # Calculate overall relevance score
        relevance_factors = {
            'category_match': category_match,
            'time_relevance': time_relevance,
            'not_generic': not is_generic,
            'has_sources': len(sources) > 0
        }
        
        # All factors must be true for high relevance
        is_highly_relevant = all(relevance_factors.values())
        
        # Calculate numerical score
        relevance_score = sum(relevance_factors.values()) / len(relevance_factors)
        
        return {
            'is_relevant': is_highly_relevant,
            'relevance_score': relevance_score,
            'category_match': category_match,
            'time_relevance': time_relevance,
            'is_generic': is_generic,
            'has_sufficient_sources': len(sources) >= 3,
            'relevance_factors': relevance_factors
        }

    def check_relevance_and_quality(self, query: str, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """Enhanced relevance and quality check"""
        try:
            # Extract query intent
            query_intent = self.extract_query_intent(query)
            
            # Validate content relevance
            relevance_check = self.validate_content_relevance(query_intent, answer, sources)
            
            # Check answer quality (length, structure, etc.)
            quality_factors = {
                'sufficient_length': len(answer) > 100,  # More strict length requirement
                'has_structure': len(answer.split('.')) > 2,  # Multiple sentences
                'not_error_message': 'error' not in answer.lower(),
                'has_specific_info': not any(phrase in answer.lower() for phrase in [
                    'i don\'t have', 'not available', 'please check', 'cannot provide'
                ])
            }
            
            quality_score = sum(quality_factors.values()) / len(quality_factors)
            is_quality = quality_score >= 0.75  # Higher threshold
            
            # Final determination
            is_sufficient = (
                relevance_check['is_relevant'] and 
                is_quality and 
                len(sources) > 0 and
                not relevance_check['is_generic']
            )
            
            return {
                'is_sufficient': is_sufficient,
                'relevance_score': relevance_check['relevance_score'],
                'quality_score': quality_score,
                'has_sources': len(sources) > 0,
                'needs_fallback': not is_sufficient,
                'query_intent': query_intent,
                'relevance_details': relevance_check,
                'quality_factors': quality_factors,
                'reason': self._get_insufficiency_reason(relevance_check, quality_factors, sources)
            }
            
        except Exception as e:
            logger.error(f"Error in relevance check: {str(e)}")
            return {
                'is_sufficient': False,
                'relevance_score': 0.0,
                'quality_score': 0.0,
                'has_sources': False,
                'needs_fallback': True,
                'reason': f'Error in verification: {str(e)}'
            }

    def _get_insufficiency_reason(self, relevance_check: Dict, quality_factors: Dict, sources: List) -> str:
        """Get specific reason why answer is insufficient"""
        reasons = []
        
        if not relevance_check['category_match']:
            reasons.append("Answer doesn't match the requested topic/category")
        
        if not relevance_check['time_relevance']:
            reasons.append("Sources are not recent enough for the time-sensitive query")
        
        if relevance_check['is_generic']:
            reasons.append("Answer is too generic or contains fallback language")
        
        if len(sources) == 0:
            reasons.append("No reliable sources found")
        
        if not quality_factors['sufficient_length']:
            reasons.append("Answer is too brief")
        
        if not quality_factors['has_specific_info']:
            reasons.append("Answer lacks specific information")
        
        return "; ".join(reasons) if reasons else "Answer meets quality standards"

    def generate_fallback_answer(self, query: str) -> str:
        """Generate a fallback answer when sources are insufficient"""
        try:
            query_intent = self.extract_query_intent(query)
            
            # Create more specific fallback based on query intent
            category_suggestions = {
                'sports': "recent sports news, match results, player updates",
                'politics': "political developments, election news, policy updates",
                'business': "market news, business developments, economic updates",
                'technology': "tech news, product launches, industry updates",
                'health': "health news, medical developments, healthcare updates",
                'entertainment': "entertainment news, celebrity updates, movie releases"
            }
            
            if query_intent['categories']:
                main_category = query_intent['categories'][0]
                suggestion = category_suggestions.get(main_category, "news in your area of interest")
                
                fallback = f"⚠️ **Limited Sources Available**\n\n"
                fallback += f"I don't have sufficient recent {main_category} news sources to provide a comprehensive answer about '{query}'. "
                fallback += f"For the most current {suggestion}, I recommend:\n\n"
                fallback += f"• Checking major {main_category} news websites\n"
                fallback += f"• Following official {main_category} social media accounts\n"
                fallback += f"• Using real-time news aggregators\n\n"
                fallback += f"Please try rephrasing your query or check back later as our news sources are updated regularly."
            else:
                fallback = f"⚠️ **Limited Sources Available**\n\n"
                fallback += f"I don't have sufficient recent news sources to answer your query about '{query}'. "
                fallback += f"For the most current information, please check:\n\n"
                fallback += f"• Major news websites\n"
                fallback += f"• Official news apps\n"
                fallback += f"• Real-time news feeds\n\n"
                fallback += f"Please try a different search term or check back later."
            
            return fallback
            
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
            # Extract query intent for better filtering
            query_intent = self.verification_agent.extract_query_intent(query)
            
            enhanced_query = self.enhance_query(query)
            docs = self.vector_store.similarity_search(enhanced_query, k=k*2)  # Get more docs initially

            # Filter by time constraints
            if any(word in query.lower() for word in ['today', 'current', 'latest']):
                docs = self.filter_recent_docs(docs, days_back=1)  # More strict for "today"
            elif 'recent' in query.lower():
                docs = self.filter_recent_docs(docs, days_back=7)

            # Filter by category if specified
            if query_intent['categories']:
                docs = self.filter_by_category(docs, query_intent['categories'])

            # Return top k documents after filtering
            filtered_docs = docs[:k]
            logger.info(f"Retrieved {len(filtered_docs)} relevant documents after filtering")
            return filtered_docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []

    def filter_by_category(self, docs: List[Document], categories: List[str]) -> List[Document]:
        """Filter documents by category keywords"""
        category_keywords = {
            'sports': ['sports', 'football', 'cricket', 'basketball', 'tennis', 'soccer', 'baseball', 'hockey', 'athletics', 'olympics', 'fifa', 'match', 'game', 'tournament', 'championship'],
            'politics': ['politics', 'election', 'government', 'minister', 'president', 'congress', 'parliament', 'policy', 'vote', 'campaign'],
            'business': ['business', 'economy', 'stock', 'market', 'finance', 'company', 'corporate', 'revenue', 'profit', 'investment', 'banking'],
            'technology': ['technology', 'tech', 'ai', 'artificial intelligence', 'software', 'hardware', 'startup', 'internet', 'cyber', 'digital'],
            'health': ['health', 'medical', 'medicine', 'hospital', 'doctor', 'disease', 'covid', 'vaccine', 'treatment', 'healthcare'],
            'entertainment': ['entertainment', 'movie', 'film', 'celebrity', 'music', 'show', 'actor', 'actress', 'hollywood', 'bollywood']
        }
        
        filtered_docs = []
        for doc in docs:
            doc_text = (doc.page_content + ' ' + doc.metadata.get('title', '')).lower()
            
            for category in categories:
                if category in category_keywords:
                    keywords = category_keywords[category]
                    if any(keyword in doc_text for keyword in keywords):
                        filtered_docs.append(doc)
                        break
        
        # If no category-specific docs found, return original docs
        return filtered_docs if filtered_docs else docs

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
                # No documents found, use verification agent for fallback
                fallback_answer = self.verification_agent.generate_fallback_answer(query)
                return {
                    'answer': fallback_answer,
                    'sources': [],
                    'query_type': query_type,
                    'is_from_sources': False,
                    'verification_status': 'no_documents_fallback_generated'
                }

            context = self.prepare_context(docs, query_type)
            initial_answer = self.generate_response(query, context, query_type)

            # Prepare sources
            sources = []
            seen_urls = set()
            for doc in docs[:8]:  # More sources for verification
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

            # Enhanced verification of the answer quality and relevance
            verification_result = self.verification_agent.check_relevance_and_quality(query, initial_answer, sources)
            
            if verification_result['is_sufficient']:
                # Answer is good, return as is
                final_answer = initial_answer
                verification_status = 'verified_from_sources'
                is_from_sources = True
            else:
                # Answer is not sufficient, generate fallback
                logger.info(f"Answer insufficient: {verification_result.get('reason', 'Unknown reason')}")
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
