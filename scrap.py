import os
import shutil
import time
import logging
import sys
import pandas as pd
from scrapy import Spider, Request
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import CloseSpider
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from transformers import pipeline, AutoTokenizer
from huggingface_hub import login
import textwrap

# ----------------- Step 0: Clean Previous Index Files -----------------
for file in ["faiss_index.db", "faiss_index.index", "faiss_index.json"]:
    if os.path.exists(file):
        os.remove(file)
if os.path.isdir("faiss_index.index"):
    shutil.rmtree("faiss_index.index")

# ----------------- Logging Setup -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Hugging Face Login -----------------
try:
    login(token="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")  # Replace with your own token
    logger.info("Logged into Hugging Face Hub.")
except Exception as e:
    logger.warning(f"Failed to log into Hugging Face: {e}")

# ----------------- Load Tokenizer -----------------
tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# ----------------- Setup Document Store -----------------
document_store = FAISSDocumentStore(
    embedding_dim=768,
    faiss_index_factory_str="Flat",
    sql_url="sqlite:///faiss_index.db"
)

# ----------------- Scrapy Spider Class -----------------
class ContentSpider(Spider):
    name = 'content_spider'
    
    def __init__(self, url=None, *args, **kwargs):
        super(ContentSpider, self).__init__(*args, **kwargs)
        self.url = url
        self.extracted_text = ""
        
    def start_requests(self):
        yield Request(url=self.url, callback=self.parse)
            
    def parse(self, response):
        # Extract all paragraph text
        paragraphs = response.css('p::text').getall()
        # Also get text from paragraph elements that might contain nested elements
        nested_paragraphs = response.css('p *::text').getall()
        
        # Combine all text
        all_text = ' '.join(paragraphs + nested_paragraphs)
        self.extracted_text = all_text.strip()

# Global variable to store scraped text
scraped_text = ""

# Custom spider with callback to save text
class TextCollectorSpider(Spider):
    name = 'text_collector'
    
    def __init__(self, url):
        self.start_urls = [url]
        
    def parse(self, response):
        global scraped_text
        paragraphs = response.css('p::text').getall()
        nested_paragraphs = response.css('p *::text').getall()
        all_text = ' '.join(paragraphs + nested_paragraphs)
        scraped_text = all_text.strip()

# ----------------- Scrape Website using Scrapy -----------------
def scrape_website(url):
    global scraped_text
    scraped_text = ""
    
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
    }
    
    # Use try-except to catch recursion errors
    try:
        process = CrawlerProcess(settings)
        process.crawl(TextCollectorSpider, url=url)
        process.start()  # This blocks until the crawl is finished
    except RecursionError:
        print("Warning: RecursionError occurred. This can happen with complex web pages.")
        print("Using a simplified approach...")
        # Fallback to a simpler approach if recursion error occurs
        import requests
        from bs4 import BeautifulSoup
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            scraped_text = ' '.join([para.get_text() for para in paragraphs if para.get_text().strip()])
        except Exception as e:
            print(f"Fallback approach also failed: {e}")
    
    return scraped_text

# ----------------- Retriever -----------------
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=False
)

# ----------------- Reader -----------------
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

# ----------------- QA Pipeline -----------------
pipe = ExtractiveQAPipeline(reader, retriever)

# ----------------- Answering Function -----------------
def answer_question(query):
    prediction = pipe.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
    answer = prediction['answers'][0].answer if prediction['answers'] else "No answer found"
    if not answer.endswith("."):
        answer += "."
    return answer.strip()

# ----------------- Save to CSV -----------------
def save_to_csv(queries, answers, filename="answers.csv"):
    df = pd.DataFrame({"Question": queries, "Answer": answers})
    df.to_csv(filename, index=False)

# ----------------- Markdown Table -----------------
def print_markdown_table(questions, answers):
    print("| No | Question | Answer |")
    print("|----|----------|--------|")
    for i, (q, a) in enumerate(zip(questions, answers), 1):
        q_clean = q.replace("|", " ")
        a_clean = a.replace("|", " ")
        print(f"| {i} | {q_clean} | {a_clean} |")

# ----------------- Question Generator -----------------
# Create the question generation pipeline
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")

def generate_questions(text, target_count=50):
    # Split the text into chunks under 512 tokens
    max_chunk_size = 500  # Keep it below model token limit
    paragraphs = textwrap.wrap(text, max_chunk_size)
    
    questions = []
    
    for para in paragraphs:
        try:
            qg_input = "generate questions: " + para
            outputs = question_generator(qg_input, max_length=64, num_return_sequences=1)
            for output in outputs:
                if output["generated_text"] not in questions:
                    questions.append(output["generated_text"])
                if len(questions) >= target_count:
                    return questions
        except Exception as e:
            print(f"Skipping chunk due to error: {e}")
            continue
            
    return questions

# ----------------- Scrape and Index -----------------
def scrape_and_index(url):
    try:
        text = scrape_website(url)
        print(f"\n✅ Scraped text from {url}...\n")
        
        if not text:
            print("Warning: No text was extracted from the URL. The page might be using JavaScript to load content, which basic Scrapy can't process by default.")
            print("You might need to use Scrapy with Splash or Playwright for JavaScript rendering.")
            return ""
            
        # Print a sample of the text to verify content was scraped correctly
        print(f"Sample text (first 200 chars): {text[:200]}...")
            
        doc = {"content": text, "meta": {"source": url}}
        document_store.write_documents([doc])
        document_store.update_embeddings(retriever)
        return text
    except Exception as e:
        print(f"Error during scraping and indexing: {e}")
        return ""

# ----------------- Domain Selection -----------------
def get_user_domain():
    print("Select a domain from the following options:")
    domains = {
        1: ("Web Scraping", "https://en.wikipedia.org/wiki/Web_scraping"),
        2: ("Machine Learning", "https://en.wikipedia.org/wiki/Machine_learning"),
        3: ("Cybersecurity", "https://en.wikipedia.org/wiki/Computer_security"),
        4: ("Healthcare", "https://en.wikipedia.org/wiki/Health_care"),
        5: ("Education", "https://en.wikipedia.org/wiki/Education")
    }
    for i, (name, _) in domains.items():
        print(f"{i}. {name}")
    choice = int(input("Enter the number corresponding to your choice: "))
    return domains.get(choice, domains[1])  # Default to Web Scraping

# ----------------- Main Execution -----------------
if __name__ == "__main__":
    try:
        # Get domain selection
        selected_domain, url = get_user_domain()
        print(f"Starting to process domain: {selected_domain} from URL: {url}")
        
        # Scrape website and index content
        text = scrape_and_index(url)
        
        if not text:
            print("Exiting due to empty text. Please check the URL or try a different one.")
            exit(1)

        # Generate Questions
        print("\nGenerating questions from the extracted text...")
        questions = generate_questions(text, target_count=50)
        if not questions:
            print("Could not generate any questions. Check if the text extraction was successful.")
            exit(1)
            
        print(f"\nGenerated {len(questions)} Questions for domain: {selected_domain}")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q}")

        # Answer Questions
        answers = []
        print("\nAnswering questions...")
        for i, query in enumerate(questions, 1):
            try:
                print(f"\nQ{i}: {query}")
                answer = answer_question(query)
                print(f"A{i}: {answer}")
                answers.append(answer)
                time.sleep(0.5)  # Reduced sleep time for faster processing
            except Exception as e:
                print(f"Error answering question {i}: {e}")
                answers.append("Error processing this question")

        # Save to CSV
        filename = f"{selected_domain.lower().replace(' ', '_')}_qa.csv"
        save_to_csv(questions, answers, filename=filename)
        print(f"\nSaved results to '{filename}'.")

        # Print Markdown Table
        print("\nQA Table:")
        print_markdown_table(questions, answers)
        print("\nDone!")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your inputs and try again.")