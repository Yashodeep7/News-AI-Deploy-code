import requests
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import pickle
import os
from datetime import datetime, timedelta
import time
import logging
from urllib.parse import urljoin, urlparse
import json
from dotenv import load_dotenv
import streamlit as st
import subprocess
import shutil

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreBuilder:
    def __init__(self):
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        if hf_token: 
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},
        )

        
        # Updated Indian news sources with RSS feeds
        self.news_sources = {
            # The Hindu
            'the_hindu': 'https://www.thehindu.com/feeder/default.rss',
            'the_hindu_national': 'https://www.thehindu.com/news/national/feeder/default.rss',
            'the_hindu_international': 'https://www.thehindu.com/news/international/feeder/default.rss',
            'the_hindu_business': 'https://www.thehindu.com/business/feeder/default.rss',
            'the_hindu_sport': 'https://www.thehindu.com/sport/feeder/default.rss',
            
            # Times of India
            'toi_india': 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms',
            'toi_world': 'https://timesofindia.indiatimes.com/rssfeeds/296589292.cms',
            'toi_business': 'https://timesofindia.indiatimes.com/rssfeeds/1898055.cms',
            'toi_sports': 'https://timesofindia.indiatimes.com/rssfeeds/4719148.cms',
            'toi_entertainment': 'https://timesofindia.indiatimes.com/rssfeeds/1081479906.cms',
            
            # Economic Times
            'et_news': 'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
            'et_markets': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
            'et_industry': 'https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms',
            'et_policy': 'https://economictimes.indiatimes.com/news/economy/policy/rssfeeds/1715249553.cms',
            'et_tech': 'https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms',

            # The Indian Express official feeds
            'indian_express_all':       'https://indianexpress.com/feed/',
            'indian_express_india':     'https://indianexpress.com/section/india/feed/',
            'indian_express_world':     'https://indianexpress.com/section/world/feed/',
            'indian_express_politics':  'https://indianexpress.com/section/politics/feed/',
            'indian_express_business':  'https://indianexpress.com/section/business/feed/',
            'indian_express_sports':    'https://indianexpress.com/section/sports/feed/',
            'indian_express_entertainment':'https://indianexpress.com/section/entertainment/feed/',
            'indian_express_technology':'https://indianexpress.com/section/technology/feed/',
            'indian_express_cities':    'https://indianexpress.com/section/cities/feed/',
            'indian_express_explained': 'https://indianexpress.com/section/explained/feed/',
            'indian_express_opinion':   'https://indianexpress.com/section/opinion/feed/',

             # International News
            'cnn_world': 'http://rss.cnn.com/rss/edition_world.rss',
            'bbc_world': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'al_jazeera': 'https://www.aljazeera.com/xml/rss/all.xml',

            # Politics
            'npr_politics': 'https://feeds.npr.org/1014/rss.xml',
            'ap_politics': 'https://apnews.com/rss/apf-politics',
            'politico': 'https://www.politico.com/rss/politics08.xml',
        
            # Indian Finance & Markets
            'mint_news': 'https://www.livemint.com/rss/news',
            'business_standard_top': 'https://www.business-standard.com/rss/home_page_top_stories.rss',
            'business_standard_sports': 'https://www.business-standard.com/rss/sports',
        
            # Global Business
            'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
        
            # Tech / Innovation
            'techcrunch': 'http://feeds.feedburner.com/TechCrunch/',
            'wired': 'https://www.wired.com/feed/rss',
        
            # Canada/Global News
            'global_news_world': 'https://globalnews.ca/world/feed/',
            'global_news_politics': 'https://globalnews.ca/politics/feed/',

            'cnn_world': 'http://rss.cnn.com/rss/edition_world.rss',
            'bbc_world': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'al_jazeera': 'https://www.aljazeera.com/xml/rss/all.xml',
            'reuters_world': 'http://feeds.reuters.com/Reuters/worldNews',
            'ap_politics': 'https://apnews.com/rss/apf-politics',
            'npr_politics': 'https://feeds.npr.org/1014/rss.xml',
            'politico': 'https://www.politico.com/rss/politics08.xml',
        
            # 📈 Global Business & Tech
            'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml',
            'techcrunch': 'http://feeds.feedburner.com/TechCrunch/',
            'wired': 'https://www.wired.com/feed/rss',
        
            # 🇮🇳 India: Business, Economy & Finance
            'mint_news': 'https://www.livemint.com/rss/news',
            'mint_politics': 'https://www.livemint.com/rss/politics',
            'financial_express': 'https://www.financialexpress.com/rss/section/economy/',
            'businessuniverse_latest': 'https://www.businessuniverse.in/rss/latest-posts',
            'businessuniverse_economy': 'https://www.businessuniverse.in/rss/category/india-news',
            'businessuniverse_stock': 'https://www.businessuniverse.in/rss/category/stock-market',
        
            # 🏏 India Sports & Entertainment

            'abp_sports': 'https://news.abplive.com/sports/feed',
            'abp_entertainment': 'https://news.abplive.com/entertainment/feed',
        
            # 🗞️ Hindi / Regional News
            'altnews': 'https://altnews.in/feed',
        
            # 🇨🇦 Canada / Global Aggregates
            'globalnews_world': 'https://globalnews.ca/world/feed/',
            'vancouver_sun': 'https://vancouversun.com/feed/?x=1',
        
            # 🌍 Misc / Other India
            'abplive_home': 'https://news.abplive.com/home/feed',
            'abplive_technology': 'https://news.abplive.com/technology/feed',
            'india_together_environment': 'https://indiatogether.org/rss_feeds_list',
                    

            'firstpost': 'https://www.firstpost.com/rss/home.xml',
            'news18': 'https://www.news18.com/rss/india.xml',
            'zee_news': 'https://zeenews.india.com/rss/india-national-news.xml',
            'hindustan_times': 'https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml',
            'business_standard': 'https://www.business-standard.com/rss/home_page_top_stories.rss'
            
        }
        
        # Updated direct websites for Indian news
        self.direct_websites = [
            'https://www.thehindu.com',
            'https://timesofindia.indiatimes.com',
            'https://economictimes.indiatimes.com',
            'https://indianexpress.com',
            'https://www.firstpost.com',
            'https://www.news18.com',
            'https://zeenews.india.com',
            'https://www.hindustantimes.com',
            'https://www.business-standard.com'
        ]
        
        # Headers for web scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def commit_and_push_to_github(self):
        """Commit and push the vector store files to GitHub repository"""
        try:
            # Get GitHub token from secrets or environment
            github_token = os.getenv("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
            github_url = os.getenv("GITHUB_URL") or st.secrets.get("GITHUB_URL")
            
            if not github_token:
                logger.warning("No GitHub token found. Files saved locally only.")
                return False
            
            # Set git config (only if not already set)
            try:
                subprocess.run(["git", "config", "user.email"], capture_output=True, check=True)
            except:
                subprocess.run(["git", "config", "user.email", "streamlit@app.com"], check=True)
            
            try:
                subprocess.run(["git", "config", "user.name"], capture_output=True, check=True)
            except:
                subprocess.run(["git", "config", "user.name", "Streamlit App"], check=True)
            
            # Add files to git
            files_to_commit = [
                "faiss_index",
                "articles_data.pickle", 
                "chunks_metadata.pickle",
                "vector_store_stats.json"
            ]
            
            for file_path in files_to_commit:
                if os.path.exists(file_path):
                    subprocess.run(["git", "add", file_path], check=True)
            
            # Check if there are changes to commit
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            if not result.stdout.strip():
                logger.info("No changes to commit")
                return True
            
            # Commit changes
            commit_message = f"Update vector store - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            
            # Push to repository using the token
            subprocess.run([
                "git", "push", 
                f"https://{github_token}@github.com/{github_url}.git", 
                "main"
            ], check=True)
            
            logger.info("Successfully pushed vector store files to GitHub")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error pushing to GitHub: {str(e)}")
            return False
        
    def scrape_rss_feeds(self):
        """Scrape news from RSS feeds"""
        all_articles = []
        
        for source_name, rss_url in self.news_sources.items():
            logger.info(f"Scraping {source_name}...")
            try:
                # Set timeout and user agent for feedparser
                feed = feedparser.parse(rss_url)
                
                if feed.bozo:
                    logger.warning(f"Feed parsing warning for {source_name}: {feed.bozo_exception}")
                
                for entry in feed.entries[:15]:  # Limit to 15 articles per source
                    try:
                        article_data = {
                            'title': entry.get('title', '').strip(),
                            'url': entry.get('link', ''),
                            'summary': entry.get('summary', '').strip(),
                            'published': entry.get('published', ''),
                            'source': source_name,
                            'content': ''
                        }
                        
                        # Skip if essential data is missing
                        if not article_data['title'] or not article_data['url']:
                            continue
                        
                        # Try to get full article content
                        full_content = self.scrape_article_content(article_data['url'])
                        if full_content:
                            article_data['content'] = full_content
                        else:
                            # Use summary if available, otherwise use title
                            article_data['content'] = article_data['summary'] or article_data['title']
                        
                        all_articles.append(article_data)
                        time.sleep(0.5)  # Be respectful to servers
                        
                    except Exception as e:
                        logger.error(f"Error processing entry from {source_name}: {str(e)}")
                        continue
                    
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {str(e)}")
                continue
        
        logger.info(f"Scraped {len(all_articles)} articles from RSS feeds")
        return all_articles

    def scrape_article_content(self, url):
        """Scrape full article content from URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'advertisement', 'ads']):
                element.decompose()
            
            # Try different selectors for article content based on Indian news sites
            content_selectors = [
                'article',
                '.article-body',
                '.story-body',
                '.entry-content',
                '.post-content',
                '.content',
                '.articlebodycontent',  # Times of India
                '.story_content',      # The Hindu
                '.Normal',             # Economic Times
                '.full-details',       # Indian Express
                '.ins_storybody',      # NDTV
                '.story-element-text', # Various Indian sites
                'main',
                '[data-testid="article-content"]'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text(strip=True) for elem in elements])
                    break
            
            # If no specific selector works, try to get all paragraphs
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
            
            # Clean up content
            content = ' '.join(content.split())  # Normalize whitespace
            content = content.replace('\n', ' ').replace('\t', ' ')
            
            # Return content if it's substantial enough
            return content if len(content) > 100 else ""
            
        except Exception as e:
            logger.error(f"Error scraping article content from {url}: {str(e)}")
            return ""

    def scrape_direct_websites(self):
        """Scrape news directly from major Indian news websites"""
        all_articles = []
        
        for website in self.direct_websites:
            logger.info(f"Scraping {website}...")
            try:
                response = requests.get(website, headers=self.headers, timeout=20)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links with improved patterns for Indian sites
                article_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    # Check for common patterns in Indian news URLs
                    if any(keyword in href.lower() for keyword in [
                        '/news/', '/article/', '/story/', '/world/', '/india/', 
                        '/business/', '/sports/', '/entertainment/', '/politics/',
                        '/economy/', '/tech/', '/lifestyle/', '/opinion/'
                    ]):
                        full_url = urljoin(website, href)
                        if full_url not in article_links and len(article_links) < 15:
                            article_links.append(full_url)
                
                # Process articles from each site
                for url in article_links[:10]:  # Limit to 10 per site
                    try:
                        article_content = self.scrape_article_content(url)
                        if article_content and len(article_content) > 200:
                            # Extract title from URL or content
                            title = self.extract_title_from_content(article_content, url)
                            
                            article_data = {
                                'title': title,
                                'url': url,
                                'content': article_content,
                                'source': urlparse(website).netloc.replace('www.', ''),
                                'published': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'summary': article_content[:300] + '...' if len(article_content) > 300 else article_content
                            }
                            all_articles.append(article_data)
                        
                        time.sleep(1)  # Be respectful
                        
                    except Exception as e:
                        logger.error(f"Error processing article {url}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error scraping {website}: {str(e)}")
                continue
        
        logger.info(f"Scraped {len(all_articles)} articles from direct websites")
        return all_articles

    def extract_title_from_content(self, content, url=""):
        """Extract a reasonable title from article content or URL"""
        try:
            # Try to extract from URL first
            if url:
                url_parts = url.split('/')
                for part in url_parts:
                    if len(part) > 10 and any(char.isalpha() for char in part):
                        # Clean up URL part to make it readable
                        title = part.replace('-', ' ').replace('_', ' ').title()
                        if len(title) > 20:
                            return title[:100] + '...' if len(title) > 100 else title
            
            # Extract from content
            sentences = content.split('.')
            if sentences:
                # Use first sentence as title, but limit length
                title = sentences[0].strip()
                return title[:100] + '...' if len(title) > 100 else title
            
            return "News Article"
        except:
            return "News Article"

    def create_chunks_with_metadata(self, articles):
        """Create chunks with proper metadata preservation"""
        documents = []
        
        for article in articles:
            try:
                # Create full article text
                full_text = f"Title: {article['title']}\n\n"
                if article.get('summary') and article['summary'] != article['content']:
                    full_text += f"Summary: {article['summary']}\n\n"
                full_text += f"Content: {article['content']}"
                
                # Split into chunks
                chunks = self.text_splitter.split_text(full_text)
                
                for i, chunk in enumerate(chunks):
                    # Create metadata for each chunk
                    metadata = {
                        'title': article['title'],
                        'url': article['url'],
                        'source': article['source'],
                        'published': article.get('published', ''),
                        'summary': article.get('summary', ''),
                        'chunk_id': f"{article['source']}_{hash(article['url'])}_{i}",
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                    
                    # Create document with chunk and metadata
                    doc = Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error creating chunks for article {article.get('title', 'Unknown')}: {str(e)}")
                continue
        
        logger.info(f"Created {len(documents)} chunks from {len(articles)} articles")
        return documents

    def build_vector_store(self):
        """Build the complete vector store"""
        try:
            logger.info("Starting vector store building process...")
            
            # Scrape articles from both RSS feeds and direct websites
            logger.info("Scraping RSS feeds...")
            articles_rss = self.scrape_rss_feeds()
            
            logger.info("Scraping direct websites...")
            articles_direct = self.scrape_direct_websites()
            
            # Combine all articles
            all_articles = articles_rss + articles_direct
            
            # Remove duplicates based on URL and title
            seen_urls = set()
            seen_titles = set()
            unique_articles = []
            
            for article in all_articles:
                url_key = article['url'].lower()
                title_key = article['title'].lower()
                
                if url_key not in seen_urls and title_key not in seen_titles:
                    seen_urls.add(url_key)
                    seen_titles.add(title_key)
                    unique_articles.append(article)
            
            logger.info(f"Total unique articles: {len(unique_articles)}")
            
            if not unique_articles:
                logger.error("No articles were scraped!")
                return False
            
            # Filter out articles with insufficient content
            filtered_articles = [
                article for article in unique_articles 
                if len(article.get('content', '')) > 50
            ]
            
            logger.info(f"Articles after content filtering: {len(filtered_articles)}")
            
            # Save raw articles data
            with open('articles_data.pickle', 'wb') as f:
                pickle.dump(filtered_articles, f)
            
            # Create chunks with metadata
            documents = self.create_chunks_with_metadata(filtered_articles)
            
            if not documents:
                logger.error("No documents created!")
                return False
            
            # Create FAISS vector store
            logger.info("Creating FAISS vector store...")
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Save vector store
            vector_store.save_local("faiss_index")
            
            # Save metadata separately for easy access
            metadata_list = [doc.metadata for doc in documents]
            with open('chunks_metadata.pickle', 'wb') as f:
                pickle.dump(metadata_list, f)
            
            logger.info("Vector store built successfully!")
            
            # Create summary statistics
            self.create_summary_stats(filtered_articles, documents)
            
            # Commit and push to GitHub
            logger.info("Pushing vector store files to GitHub...")
            github_success = self.commit_and_push_to_github()
            
            if github_success:
                logger.info("Vector store files successfully pushed to GitHub!")
            else:
                logger.warning("Files saved locally but not pushed to GitHub")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building vector store: {str(e)}")
            return False

    def create_summary_stats(self, articles, documents):
        """Create summary statistics"""
        source_counts = {}
        for article in articles:
            source = article['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        stats = {
            'total_articles': len(articles),
            'total_chunks': len(documents),
            'sources': list(source_counts.keys()),
            'source_counts': source_counts,
            'build_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'avg_chunks_per_article': len(documents) / len(articles) if articles else 0,
            'avg_content_length': sum(len(article.get('content', '')) for article in articles) / len(articles) if articles else 0
        }
        
        with open('vector_store_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics: {stats}")
        print(f"\n=== BUILD SUMMARY ===")
        print(f"Total Articles: {stats['total_articles']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Sources: {len(stats['sources'])}")
        print(f"Articles per source: {stats['source_counts']}")
        print(f"Average chunks per article: {stats['avg_chunks_per_article']:.2f}")
        print(f"Build completed at: {stats['build_time']}")

if __name__ == "__main__":
    builder = VectorStoreBuilder()
    success = builder.build_vector_store()
    if success:
        print("\n✅ Vector store built successfully!")
        print("Files created:")
        print("- faiss_index/ (vector store)")
        print("- articles_data.pickle (raw articles)")
        print("- chunks_metadata.pickle (chunk metadata)")
        print("- vector_store_stats.json (statistics)")
    else:
        print("\n❌ Failed to build vector store!")
