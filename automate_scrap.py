import os
import sys
import time
import logging
import re
import json
import pandas as pd
import argparse
from urllib.parse import urlparse
from scrapy import Spider, Request
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import CloseSpider
from transformers import pipeline, AutoTokenizer
from huggingface_hub import login
import textwrap
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')



# ----------------- Logging Setup -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qa_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------- Download NLTK resources -----------------
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {e}")

# ----------------- Ad Content Detection Patterns -----------------
AD_PATTERNS = [
    r'advertisement',
    r'sponsored',
    r'ad\s',
    r'\sad\s',
    r'promotion',
    r'subscribe now',
    r'sign up today',
    r'limited time offer',
    r'special offer',
    r'buy now',
    r'click here',
    r'newsletter',
    r'cookie',
    r'privacy policy',
    r'terms of service',
    r'download now',
    r'free trial',
    r'sale',
    r'discount',
    r'advert'
]

AD_CLASSES = [
    'ad', 'ads', 'advert', 'advertisement', 'banner', 'promo', 'promotion', 'sponsored',
    'sidebar', 'footer', 'header', 'cookie', 'popup', 'modal', 'newsletter'
]

AD_IDS = [
    'ad', 'ads', 'advert', 'advertisement', 'banner', 'promo', 'promotion', 'sponsored',
    'sidebar', 'cookie-banner', 'newsletter-signup', 'popup', 'modal'
]

# ----------------- Scrapy Spider Classes -----------------
class ContentExtractor(Spider):
    name = 'content_extractor'
    
    def __init__(self, url, max_pages=10, *args, **kwargs):
        super(ContentExtractor, self).__init__(*args, **kwargs)
        self.start_urls = [url]
        self.allowed_domains = [urlparse(url).netloc]
        self.max_pages = int(max_pages)
        self.visited_pages = 0
        self.collected_data = []
        self.visited_urls = set()
        
    def parse(self, response):
        # Skip if we've reached max pages or already visited this URL
        if self.visited_pages >= self.max_pages or response.url in self.visited_urls:
            return
        
        self.visited_pages += 1
        self.visited_urls.add(response.url)
        
        # Extract page title
        title = response.css('title::text').get() or ""
        
        # Extract main content, avoiding ads and navigation elements
        main_content = self.extract_main_content(response)
        
        if main_content:
            self.collected_data.append({
                'url': response.url,
                'title': title.strip(),
                'content': main_content
            })
            logger.info(f"Extracted content from {response.url}")
        
        # Follow links within the same domain for additional pages
        if self.visited_pages < self.max_pages:
            internal_links = response.css('a::attr(href)').getall()
            for link in internal_links:
                if link and not self.is_likely_ad_link(link):
                    yield response.follow(link, self.parse)
    
    def extract_main_content(self, response):
        """Extract main content while filtering out ads and irrelevant content"""
        
        # Try to identify main content containers first
        main_selectors = [
            'main',
            'article',
            '.main-content',
            '#main-content',
            '.post-content',
            '.entry-content',
            '#content',
            '.content',
            '.page-content',
            '.article-content'
        ]
        
        content = ""
        
        # Try to find content in main content areas
        for selector in main_selectors:
            elements = response.css(f'{selector}')
            if elements:
                for element in elements:
                    # Get all paragraph text
                    paragraphs = element.css('p::text, p *::text').getall()
                    filtered_paragraphs = [p for p in paragraphs if self.is_valid_content(p)]
                    if filtered_paragraphs:
                        content += ' '.join(filtered_paragraphs) + ' '
        
        # If no content found in main areas, fall back to all paragraphs
        if not content.strip():
            # Filter out elements that are likely ads
            for ad_class in AD_CLASSES:
                response.css(f'.{ad_class}').remove()
            for ad_id in AD_IDS:
                response.css(f'#{ad_id}').remove()
            
            # Get remaining paragraphs
            paragraphs = response.css('p::text, p *::text').getall()
            filtered_paragraphs = [p for p in paragraphs if self.is_valid_content(p)]
            content = ' '.join(filtered_paragraphs)
        
        return content.strip()
    
    def is_valid_content(self, text):
        """Check if text is likely to be valid content rather than ads"""
        text = text.strip()
        
        # Skip empty strings
        if not text:
            return False
            
        # Skip very short strings (likely not meaningful content)
        if len(text) < 15:
            return False
            
        # Check against ad patterns
        for pattern in AD_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False
                
        return True
    
    def is_likely_ad_link(self, link):
        """Check if a link is likely to be an advertisement"""
        for pattern in AD_PATTERNS:
            if re.search(pattern, link, re.IGNORECASE):
                return True
                
        return False

# Global variable to store extracted content
extracted_content = []

# ----------------- Scrape Website using Scrapy -----------------
def scrape_website(url, max_pages=10):
    global extracted_content
    extracted_content = []
    
    # Configure Scrapy settings
    settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'ERROR',  # Reduce logging
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 1,  # Be gentle
        'DOWNLOAD_TIMEOUT': 30,    # Timeout in seconds
        'RETRY_TIMES': 2,          # Number of retries
        'COOKIES_ENABLED': True,   # Enable cookies
        'HTTPCACHE_ENABLED': True, # Enable caching
        # Set recursion limit to avoid RecursionError
        'RECURSION_LIMIT': 5000,
        # Disable redirect middleware to avoid circular redirects
        'REDIRECT_ENABLED': False
    }
    
    try:
        process = CrawlerProcess(settings)
        spider = ContentExtractor(url=url, max_pages=max_pages)
        process.crawl(spider)
        process.start()  # This blocks until the crawl is finished
        
        # Get collected data from the spider
        extracted_content = spider.collected_data
        
    except Exception as e:
        logger.error(f"Error in Scrapy crawling: {e}")
        # Fallback to a simpler approach if Scrapy fails
        try:
            logger.info(f"Falling back to requests + BeautifulSoup for {url}")
            extracted_content = fallback_scraper(url)
        except Exception as fallback_error:
            logger.error(f"Fallback scraping also failed: {fallback_error}")
    
    return extracted_content

# ----------------- Fallback Scraper -----------------
def fallback_scraper(url):
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove potential ad elements
        for ad_class in AD_CLASSES:
            for element in soup.find_all(class_=lambda c: c and ad_class in c.lower()):
                element.decompose()
                
        for ad_id in AD_IDS:
            for element in soup.find_all(id=lambda i: i and ad_id in i.lower()):
                element.decompose()
        
        # Get title
        title = soup.title.string if soup.title else ""
        
        # Try to find main content
        main_content = None
        for tag in ['main', 'article', 'div']:
            for attr in ['id', 'class']:
                for content_indicator in ['content', 'main', 'article', 'post', 'entry']:
                    elements = soup.find_all(tag, {attr: lambda x: x and content_indicator in x.lower()})
                    if elements:
                        main_content = elements[0]
                        break
                if main_content:
                    break
            if main_content:
                break
        
        # If no main content identified, use the whole body
        if not main_content:
            main_content = soup.body
        
        # Extract paragraphs from main content
        paragraphs = []
        if main_content:
            for p in main_content.find_all('p'):
                text = p.get_text().strip()
                if text and len(text) > 15:
                    # Check if text matches any ad pattern
                    is_ad = False
                    for pattern in AD_PATTERNS:
                        if re.search(pattern, text, re.IGNORECASE):
                            is_ad = True
                            break
                    if not is_ad:
                        paragraphs.append(text)
        
        content = ' '.join(paragraphs)
        
        return [{
            'url': url,
            'title': title.strip(),
        }]  
        
    except Exception as e:
        logger.error(f"Fallback scraper error: {e}")
        return []

# ----------------- Text Preprocessing -----------------
def preprocess_text(text):
    """Clean and preprocess the extracted text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep sentence punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', ' ', text)
    
    return text

# ----------------- Question Generation -----------------
def generate_questions(text, num_questions=200):
    """Generate questions from the extracted text"""
    logger.info(f"Initializing question generation model...")
    
    try:
        # Create the question generation pipeline
        question_generator = pipeline(
            "text2text-generation", 
            model="valhalla/t5-base-qa-qg-hl", 
            max_length=64
        )
        
        # Split the text into sentences
        sentences = sent_tokenize(text)
        
        # Group sentences into chunks to avoid token limits
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 500:  # Keep chunks under 500 chars
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:  # Add the last chunk
            chunks.append(current_chunk.strip())
        
        # Generate questions for each chunk
        all_questions = []
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
                
            logger.info(f"Generating questions for chunk {i+1}/{len(chunks)}...")
            
            try:
                qg_input = "generate questions: " + chunk
                outputs = question_generator(
                    qg_input, 
                    max_length=64, 
                    num_return_sequences=min(5, num_questions - len(all_questions))
                )
                
                for output in outputs:
                    question = output["generated_text"].strip()
                    if question and question not in all_questions:
                        all_questions.append(question)
                        
                # If we've generated enough questions, stop
                if len(all_questions) >= num_questions:
                    break
                    
            except Exception as e:
                logger.error(f"Error generating questions for chunk {i+1}: {e}")
                continue
        
        logger.info(f"Generated {len(all_questions)} questions")
        return all_questions
        
    except Exception as e:
        logger.error(f"Failed to initialize question generation model: {e}")
        return []

# ----------------- Answer Generation -----------------
def generate_answers(questions, text):
    """Generate answers for the given questions using the extracted text"""
    logger.info(f"Initializing answer generation model...")
    
    try:
        # Create the answer generation pipeline
        answer_generator = pipeline(
            "question-answering", 
            model="deepset/roberta-base-squad2"
        )
        
        # Generate answers for each question
        answers = []
        for i, question in enumerate(questions):
            logger.info(f"Generating answer for question {i+1}/{len(questions)}")
            
            try:
                result = answer_generator(
                    question=question,
                    context=text,
                    max_answer_len=200
                )
                
                answer = result.get("answer", "").strip()
                if not answer:
                    answer = "No answer found in the text."
                    
                answers.append(answer)
                
            except Exception as e:
                logger.error(f"Error generating answer for question {i+1}: {e}")
                answers.append("Error generating answer for this question.")
                
            # Add a small delay to avoid overwhelming the model
            time.sleep(0.2)
        
        logger.info(f"Generated {len(answers)} answers")
        return answers
        
    except Exception as e:
        logger.error(f"Failed to initialize answer generation model: {e}")
        return ["Error generating answers"] * len(questions)

# ----------------- Save Results -----------------
def save_results(url, questions, answers, output_file=None):
    """Save the results to a CSV file"""
    if not output_file:
        # Generate filename from the domain name
        domain = urlparse(url).netloc.replace("www.", "")
        output_file = f"{domain.replace('.', '_')}_qa_dataset.csv"
    
    df = pd.DataFrame({
        "question": questions,
        "answer": answers
    })
    
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(questions)} QA pairs to {output_file}")
    
    return output_file

# ----------------- Print Markdown Table -----------------
def print_markdown_table(questions, answers, limit=10):
    """Print a markdown table with sample QA pairs"""
    # Limit the display to specified number of results
    sample_questions = questions[:limit]
    sample_answers = answers[:limit]
    
    print("\n| No | Question | Answer |")
    print("|-----|----------|--------|")
    
    for i, (q, a) in enumerate(zip(sample_questions, sample_answers), 1):
        # Clean and truncate for better display
        q_clean = q.replace("|", "-").replace("\n", " ")
        a_clean = a.replace("|", "-").replace("\n", " ")
        
        # Truncate long answers
        if len(a_clean) > 100:
            a_clean = a_clean[:97] + "..."
            
        print(f"| {i} | {q_clean} | {a_clean} |")

# ----------------- Main Function -----------------
def main():
    parser = argparse.ArgumentParser(description='Scrape a website and generate QA pairs')
    parser.add_argument('url', type=str, help='URL to scrape')
    parser.add_argument('--pages', type=int, default=5, help='Maximum number of pages to scrape (default: 5)')
    parser.add_argument('--questions', type=int, default=150, help='Number of questions to generate (default: 150)')
    parser.add_argument('--output', type=str, help='Output CSV file name (optional)')
    
    args = parser.parse_args()
    
    try:
        # Step 1: Scrape the website
        logger.info(f"Scraping website: {args.url}")
        scraped_data = scrape_website(args.url, max_pages=args.pages)
        
        if not scraped_data:
            logger.error("No content extracted from the website")
            return
        
        # Step 2: Combine all content
        all_content = ""
        for item in scraped_data:
            all_content += item['content'] + " "
        
        # Step 3: Preprocess the text
        processed_text = preprocess_text(all_content)
        logger.info(f"Extracted and processed {len(processed_text)} characters of text")
        
        # Step 4: Generate questions
        logger.info(f"Generating {args.questions} questions...")
        questions = generate_questions(processed_text, num_questions=args.questions)
        
        if not questions:
            logger.error("Failed to generate any questions")
            return
        
        # Step 5: Generate answers
        logger.info(f"Generating answers for {len(questions)} questions...")
        answers = generate_answers(questions, processed_text)
        
        # Step 6: Save results to CSV
        output_file = save_results(args.url, questions, answers, args.output)
        
        # Step 7: Print a sample of the results
        print(f"\nGenerated {len(questions)} QA pairs from {args.url}")
        print(f"Results saved to {output_file}")
        print("\nSample of generated QA pairs:") 
        print_markdown_table(questions, answers)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
    