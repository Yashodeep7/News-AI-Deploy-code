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
import random
from urllib.robotparser import RobotFileParser

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
            'indian_express_all': 'https://indianexpress.com/feed/',
            'indian_express_india': 'https://indianexpress.com/section/india/feed/',
            'indian_express_world': 'https://indianexpress.com/section/world/feed/',
            'indian_express_politics': 'https://indianexpress.com/section/politics/feed/',
            'indian_express_business': 'https://indianexpress.com/section/business/feed/',
            'indian_express_sports': 'https://indianexpress.com/section/sports/feed/',
            'indian_express_entertainment': 'https://indianexpress.com/section/entertainment/feed/',
            'indian_express_technology': 'https://indianexpress.com/section/technology/feed/',
            'indian_express_cities': 'https://indianexpress.com/section/cities/feed/',
            'indian_express_explained': 'https://indianexpress.com/section/explained/feed/',
            'indian_express_opinion': 'https://indianexpress.com/section/opinion/feed/',

            # International News
            'cnn_world': 'http://rss.cnn.com/rss/edition_world.rss',
            'bbc_world': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'al_jazeera': 'https://www.aljazeera.com/xml/rss/all.xml',
            'reuters_world': 'http://feeds.reuters.com/Reuters/worldNews',

            # Politics
            'npr_politics': 'https://feeds.npr.org/1014/rss.xml',
            'ap_politics': 'https://apnews.com/rss/apf-politics',
            'politico': 'https://www.politico.com/rss/politics08.xml',
        
            # Indian Finance & Markets
            'mint_news': 'https://www.livemint.com/rss/news',
            'mint_politics': 'https://www.livemint.com/rss/politics',
            'business_standard_top': 'https://www.business-standard.com/rss/home_page_top_stories.rss',
            'business_standard_sports': 'https://www.business-standard.com/rss/sports',
            'financial_express': 'https://www.financialexpress.com/rss/section/economy/',
        
            # Tech / Innovation
            'techcrunch': 'http://feeds.feedburner.com/TechCrunch/',
            'wired': 'https://www.wired.com/feed/rss',
        
            # Global Business
            'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml',
            
            # Additional sources that are more scraping-friendly  
            'firstpost': 'https://www.firstpost.com/rss/home.xml',
            'news18': 'https://www.news18.com/rss/india.xml',
            'hindustan_times': 'https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml',
        }
        
        # Multiple user agents to rotate
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36'
        ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def get_headers(self, referer=None):
        """Get random headers for requests"""
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Cache-Control': 'max-age=0'
        }
        
        if referer:
            headers['Referer'] = referer
            
        return headers

    def check_robots_txt(self, url):
        """Check if URL is allowed by robots.txt"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            return rp.can_fetch('*', url)
        except:
            return True  # If we can't check, assume it's allowed

    def make_request_with_retry(self, url, max_retries=3, delay_range=(2, 5)):
        """Make HTTP request with retry logic and different headers"""
        for attempt in range(max_retries):
            try:
                # Random delay to avoid being flagged as bot
                if attempt > 0:
                    delay = random.uniform(*delay_range)
                    time.sleep(delay)
                
                # Use different headers for each attempt
                headers = self.get_headers(referer=urlparse(url).netloc)
                
                # Add some randomization to request timing
                response = requests.get(
                    url, 
                    headers=headers, 
                    timeout=20,
                    allow_redirects=True,
                    verify=True
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 403:
                    logger.warning(f"403 Forbidden for {url} on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                elif response.status_code == 429:
                    logger.warning(f"Rate limited for {url}, waiting longer...")
                    time.sleep(random.uniform(10, 20))
                    continue
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for {url} on attempt {attempt + 1}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {url}: {str(e)}")
            
            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))
        
        return None

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
        """Scrape news from RSS feeds with improved error handling"""
        all_articles = []
        successful_sources = 0
        failed_sources = 0
        
        for source_name, rss_url in self.news_sources.items():
            logger.info(f"Scraping {source_name}...")
            try:
                # Use requests to fetch RSS with custom headers
                response = self.make_request_with_retry(rss_url)
                if not response:
                    logger.error(f"Failed to fetch RSS feed for {source_name}")
                    failed_sources += 1
                    continue
                
                # Parse the RSS feed
                feed = feedparser.parse(response.content)
                
                if feed.bozo:
                    logger.warning(f"Feed parsing warning for {source_name}: {feed.bozo_exception}")
                
                articles_from_source = 0
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
                        
                        # Clean URL (remove tracking parameters)
                        clean_url = article_data['url'].split('#')[0].split('?')[0]
                        article_data['url'] = clean_url
                        
                        # Try to get full article content with better error handling
                        full_content = self.scrape_article_content_safe(article_data['url'], source_name)
                        if full_content:
                            article_data['content'] = full_content
                        else:
                            # Use summary if available, otherwise use title
                            article_data['content'] = article_data['summary'] or article_data['title']
                        
                        # Only add if we have substantial content
                        if len(article_data['content']) > 50:
                            all_articles.append(article_data)
                            articles_from_source += 1
                        
                        # Random delay between articles
                        time.sleep(random.uniform(0.5, 2.0))
                        
                    except Exception as e:
                        logger.error(f"Error processing entry from {source_name}: {str(e)}")
                        continue
                
                if articles_from_source > 0:
                    successful_sources += 1
                    logger.info(f"Successfully scraped {articles_from_source} articles from {source_name}")
                else:
                    failed_sources += 1
                    logger.warning(f"No articles scraped from {source_name}")
                    
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {str(e)}")
                failed_sources += 1
                continue
        
        logger.info(f"RSS Scraping completed: {successful_sources} successful, {failed_sources} failed sources")
        logger.info(f"Total articles from RSS feeds: {len(all_articles)}")
        return all_articles

    def scrape_article_content_safe(self, url, source_name):
        """Safely scrape article content with better error handling"""
        try:
            # Skip if URL seems problematic
            if '#publisher=newsstand' in url or 'utm_' in url:
                # Clean the URL
                url = url.split('#')[0].split('?')[0]
            
            # Check robots.txt (optional, can be commented out if causing issues)
            # if not self.check_robots_txt(url):
            #     logger.info(f"Robots.txt disallows scraping {url}")
            #     return ""
            
            response = self.make_request_with_retry(url)
            if not response:
                return ""
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 
                               'advertisement', 'ads', '.ad', '.advertisement', 
                               '.social-share', '.related-articles', '.comments']):
                element.decompose()
            
            # Try different selectors for article content based on source
            content_selectors = self.get_content_selectors_for_source(source_name)
            
            content = ""
            for selector in content_selectors:
                try:
                    elements = soup.select(selector)
                    if elements:
                        content = ' '.join([elem.get_text(strip=True) for elem in elements])
                        if len(content) > 100:  # Only use if substantial content
                            break
                except Exception:
                    continue
            
            # If no specific selector works, try to get all paragraphs
            if not content or len(content) < 100:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs 
                                  if len(p.get_text(strip=True)) > 20])
            
            # Clean up content
            content = ' '.join(content.split())  # Normalize whitespace
            content = content.replace('\n', ' ').replace('\t', ' ')
            
            # Return content if it's substantial enough
            return content if len(content) > 100 else ""
            
        except Exception as e:
            logger.error(f"Error scraping article content from {url}: {str(e)}")
            return ""

    def get_content_selectors_for_source(self, source_name):
        """Get content selectors based on source"""
        selectors_map = {
            'the_hindu': ['.story-content', '.story-element-text', 'article'],
            'toi': ['.Normal', '.articlebodycontent', 'article'],
            'et': ['.Normal', '.article-body', 'article'],
            'indian_express': ['.full-details', '.story-element-text', 'article'],
            'ndtv': ['.ins_storybody', '.story-element-text', 'article'],
            'mint': ['.story-body', '.article-body', 'article'],
            'firstpost': ['.story-element-text', 'article'],
            'news18': ['.story-element-text', 'article'],
            'hindustan_times': ['.story-element-text', '.story-body', 'article']
        }
        
        # Get specific selectors for source or use default
        for key in selectors_map:
            if key in source_name:
                return selectors_map[key] + self.get_default_selectors()
        
        return self.get_default_selectors()

    def get_default_selectors(self):
        """Get default content selectors"""
        return [
            'article',
            '.article-body',
            '.story-body', 
            '.entry-content',
            '.post-content',
            '.content',
            '.story-element-text',
            'main',
            '[data-testid="article-content"]',
            '.article-content'
        ]

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
        """Build the complete vector store with improved error handling"""
        try:
            logger.info("Starting vector store building process...")
            
            # Only scrape RSS feeds for now (more reliable than direct website scraping)
            logger.info("Scraping RSS feeds...")
            all_articles = self.scrape_rss_feeds()
            
            if not all_articles:
                logger.error("No articles were scraped!")
                return False
            
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
            
            # Filter out articles with insufficient content
            filtered_articles = [
                article for article in unique_articles 
                if len(article.get('content', '')) > 50
            ]
            
            logger.info(f"Articles after content filtering: {len(filtered_articles)}")
            
            if not filtered_articles:
                logger.error("No articles with sufficient content!")
                return False
            
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
