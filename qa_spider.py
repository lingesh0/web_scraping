import scrapy

class WikipediaSpider(scrapy.Spider):
    name = "wikipedia_spider"

    def __init__(self, query=None, **kwargs):
        super().__init__(**kwargs)
        self.query = query

    def start_requests(self):
        url = f"https://en.wikipedia.org/wiki/{self.query.replace(' ', '_')}"
        yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        paragraphs = response.css("p::text").getall()
        content = " ".join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())
        yield {"content": content}
