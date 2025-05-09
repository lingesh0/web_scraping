import os
import shutil
import time
import logging
import pandas as pd
import re
from typing import List, Dict, Any

import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from scrapy import signals
from twisted.internet import reactor, defer

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from huggingface_hub import login

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global list to collect content from spider
scraped_data = []

# Define Scrapy Spider with enhanced content extraction
class ContentSpider(scrapy.Spider):
    name = "content_spider"
    
    def __init__(self, url=None, *args, **kwargs):
        super(ContentSpider, self).__init__(*args, **kwargs)
        self.start_urls = [url] if url else ["https://en.wikipedia.org/wiki/Web_scraping"]

    def parse(self, response):
        # Extract content from multiple HTML elements for comprehensive coverage
        content_parts = []
        
        # Get title and headers
        title = ' '.join(response.css('title::text').getall())
        h1s = ' '.join(response.css('h1::text').getall())
        h2s = ' '.join(response.css('h2::text').getall())
        h3s = ' '.join(response.css('h3::text').getall())
        
        # Get paragraphs with nested elements
        paragraphs = ' '.join(response.css('p::text').getall())
        paragraphs += ' '.join(response.css('p *::text').getall())
        
        # Get list items
        list_items = ' '.join(response.css('li::text').getall())
        list_items += ' '.join(response.css('li *::text').getall())
        
        # Get table content
        table_content = ' '.join(response.css('table::text').getall())
        table_content += ' '.join(response.css('table *::text').getall())
        
        # Get div content that might contain important information
        div_content = ' '.join(response.css('div.content::text, div.main::text, div.article::text').getall())
        div_content += ' '.join(response.css('div.content *::text, div.main *::text, div.article *::text').getall())
        
        # Combine all content
        content_parts = [title, h1s, h2s, h3s, paragraphs, list_items, table_content, div_content]
        content = ' '.join([part for part in content_parts if part.strip()])
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Create item
        item = {
            "content": content,
            "meta": {
                "source": response.url,
                "title": title.strip()
            }
        }
        
        # Store scraped data
        scraped_data.append(item)
        
        # Log successful scraping
        logger.info(f"Successfully scraped content from {response.url} ({len(content)} chars)")
        
        yield item


class WebScrapingQASystem:
    def __init__(self):
        self._clean_previous_index()
        
        try:
            login(token="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")  # Replace with your token
            logger.info("Logged into Hugging Face Hub.")
        except Exception as e:
            logger.warning(f"Failed to log into Hugging Face: {e}")
        
        self.document_store = FAISSDocumentStore(
            embedding_dim=768,
            faiss_index_factory_str="Flat",
            sql_url="sqlite:///faiss_index.db"
        )

        self.retriever = DensePassageRetriever(
            document_store=self.document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            use_gpu=False
        )

        self.reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)

        # Initialize QA pipeline with increased top_k
        self.pipe = ExtractiveQAPipeline(self.reader, self.retriever)

        # Initialize question generation
        self.tokenizer_qg = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap", use_fast=False)
        self.model_qg = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
        self.qg_pipeline = pipeline("text2text-generation", model=self.model_qg, tokenizer=self.tokenizer_qg)
        
        # Initialize text summarizer for better question generation
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            logger.info("Text summarizer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize summarizer: {e}")
            self.summarizer = None

    def _clean_previous_index(self):
        for file in ["faiss_index.db", "faiss_index.index", "faiss_index.json"]:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Removed old index file: {file}")
        
        if os.path.isdir("faiss_index.index"):
            shutil.rmtree("faiss_index.index")
            logger.info("Removed old index directory")

    def scrape_with_scrapy(self, url):
        """Run scrapy spider asynchronously and return collected data."""
        global scraped_data
        scraped_data = []  # Reset the global list
        
        configure_logging()
        runner = CrawlerRunner()

        @defer.inlineCallbacks
        def crawl():
            yield runner.crawl(ContentSpider, url=url)
            reactor.stop()

        # Start crawler and reactor
        logger.info(f"Starting scrapy crawler for {url}")
        crawl()
        reactor.run()  # This will block until the crawler is finished

        logger.info(f"Scraped {len(scraped_data)} items from {url}")
        return scraped_data

    def index_documents(self, documents):
        if not documents:
            logger.warning("No documents to index")
            return False
        try:
            self.document_store.write_documents(documents)
            logger.info("Updating document embeddings...")
            self.document_store.update_embeddings(self.retriever)
            logger.info("Finished updating embeddings.")
            return True
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False

    def extract_key_sentences(self, text, max_sentences=10):
        """Extract key sentences for better question generation"""
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short sentences and sentences without alphabetical chars
        sentences = [s for s in sentences if len(s) > 20 and re.search(r'[a-zA-Z]', s)]
        
        # Use summarizer if available, otherwise select first sentences
        if self.summarizer and len(text) > 100:
            try:
                summary = self.summarizer(text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
                summary_sentences = re.split(r'(?<=[.!?])\s+', summary)
                return summary_sentences[:max_sentences]
            except Exception as e:
                logger.warning(f"Summarization failed: {e}")
        
        # Return first N sentences if no summarizer or summarization failed
        return sentences[:max_sentences]

    def generate_domain_aware_questions(self, text, url, max_questions=8):
        """Generate questions that are aware of the domain content"""
        try:
            # Extract domain from URL for context
            domain_parts = url.split('//')[-1].split('/')
            domain = domain_parts[0]
            topic = domain_parts[-1].replace('_', ' ') if len(domain_parts) > 1 else ""
            
            # Get key sentences for better questions
            key_sentences = self.extract_key_sentences(text)
            
            all_questions = []
            
            # Generate general domain questions
            if topic:
                domain_prompts = [
                    f"generate questions about {topic}",
                    f"what is {topic}",
                    f"why is {topic} important"
                ]
                
                for prompt in domain_prompts:
                    try:
                        results = self.qg_pipeline(prompt, max_length=64, num_return_sequences=2)
                        all_questions.extend([res["generated_text"] for res in results])
                    except Exception as e:
                        logger.warning(f"Failed to generate domain questions: {e}")
            
            # Generate content-specific questions from key sentences
            for sentence in key_sentences[:5]:  # Use top 5 sentences
                try:
                    prompt = "generate questions: " + sentence
                    results = self.qg_pipeline(prompt, max_length=64, num_return_sequences=1)
                    all_questions.extend([res["generated_text"] for res in results])
                except Exception as e:
                    logger.warning(f"Failed on sentence: {e}")
            
            # Filter questions to ensure quality and uniqueness
            filtered_questions = []
            seen = set()
            
            for q in all_questions:
                q = q.strip()
                q_lower = q.lower()
                
                # Apply filtering criteria
                if (
                    len(q) > 10 and 
                    q_lower not in seen and 
                    not q_lower.startswith("generate") and
                    "?" in q
                ):
                    seen.add(q_lower)
                    filtered_questions.append(q)
            
            # Ensure we don't exceed max_questions
            filtered_questions = filtered_questions[:max_questions]
            
            logger.info(f"Generated {len(filtered_questions)} domain-aware questions")
            return filtered_questions
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return []

    def answer_question(self, query):
        try:
            # Use increased top_k for both retriever and reader
            prediction = self.pipe.run(
                query=query, 
                params={
                    "Retriever": {"top_k": 5},  # Retrieve more documents
                    "Reader": {"top_k": 3}      # Return top 3 answers instead of just 1
                }
            )
            
            # If there are multiple answers, combine them
            if prediction['answers'] and len(prediction['answers']) > 1:
                # Get the top 3 answers (or fewer if there are less)
                top_answers = prediction['answers'][:3]
                
                # Check if we have a high confidence answer
                best_answer = top_answers[0]
                if best_answer.score > 0.8:
                    return best_answer.answer
                
                # Otherwise, combine answers if they're different
                unique_answers = set(a.answer for a in top_answers)
                if len(unique_answers) > 1:
                    # Return answers with their confidence scores
                    combined = "\n".join([
                        f"{a.answer} (confidence: {a.score:.2f})" 
                        for a in top_answers
                    ])
                    return combined
                
            # Default to the best answer or "No answer found"
            return prediction['answers'][0].answer if prediction['answers'] else "No answer found"
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error answering question: {str(e)}"

    def process_queries(self, queries):
        results = {}
        for query in queries:
            logger.info(f"Processing query: {query}")
            answer = self.answer_question(query)
            results[query] = answer
            time.sleep(0.5)
        return results

    def save_results_to_csv(self, results, filename="qa_results.csv"):
        df = pd.DataFrame({"Query": list(results.keys()), "Answer": list(results.values())})
        df.to_csv(filename, index=False)
        logger.info(f"Saved results to {filename}")


def main():
    qa_system = WebScrapingQASystem()

    # URL to scrape
    url = "https://en.wikipedia.org/wiki/Web_scraping"
    
    # Scrape website using enhanced spider
    documents = qa_system.scrape_with_scrapy(url)

    if not documents:
        logger.error("Failed to scrape any content.")
        return

    # Index documents
    success = qa_system.index_documents(documents)
    if not success:
        logger.error("Failed to index documents.")
        return

    # Extract full text for question generation
    full_text = documents[0]["content"] if documents else ""

    # Generate domain-aware questions
    auto_queries = qa_system.generate_domain_aware_questions(full_text, url, max_questions=8)

    if auto_queries:
        # Process auto-generated domain-aware questions
        auto_results = qa_system.process_queries(auto_queries)
        qa_system.save_results_to_csv(auto_results, "auto_qa_results.csv")

        print("\n=== Auto-Generated Domain-Aware Questions and Answers ===")
        for query, answer in auto_results.items():
            print(f"Q: {query}")
            print(f"A: {answer}\n")
    else:
        logger.warning("No auto-generated questions.")

    # Define custom queries
    custom_queries = [
        "What is web scraping?",
        "Are there legal concerns with web scraping?",
        "What are some common tools for web scraping?",
        "How can websites protect against unwanted scraping?",
        "What are the ethical considerations in web scraping?"
    ]

    # Process custom queries
    custom_results = qa_system.process_queries(custom_queries)
    qa_system.save_results_to_csv(custom_results, "custom_qa_results.csv")

    print("\n=== Custom Questions and Answers ===")
    for query, answer in custom_results.items():
        print(f"Q: {query}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
