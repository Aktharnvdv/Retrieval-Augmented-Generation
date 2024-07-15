import scrapy
import json
from urllib.parse import urljoin, urlparse, urldefrag
from scrapy.crawler import CrawlerProcess
from scrapy.selector import Selector
from multiprocessing import Process

class MySpider(scrapy.Spider):
    name = 'MySpider'

    def __init__(self, start_url=None, domain=None, depth=1, *args, **kwargs):
        """Initializes the spider with the starting URL, domain, and depth limit.

        Args:
            start_url (str): The URL to start scraping from.
            domain (str): The domain to allow for scraping.
            depth (int): The maximum depth of the crawl.
        """
        super(MySpider, self).__init__(*args, **kwargs)
        self.start_url = start_url
        self.allowed_domains = [domain]
        self.depth_limit = int(depth)
        self.scraped_items = []
        self.non_text_file_types = ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.doc', '.docx', '.xls', '.xlsx']
        self.page_count = 0
        self.max_pages = 50
        self.visited_urls = set()

    def start_requests(self):
        """Starts the scraping process by making the initial request."""
        yield scrapy.Request(url=self.start_url, callback=self.parse, meta={'depth': 0})

    def parse(self, response):
        """Parses the response from the server and extracts links and text.

        Args:
            response (scrapy.http.Response): The response object from the request.
        """
        if self.page_count >= self.max_pages:
            self.crawler.engine.close_spider(self, reason='page limit reached')
            return

        depth = response.meta['depth']
        parent_url = response.request.url
        content_type = response.headers.get('Content-Type', b'').decode('utf-8').lower()

        if depth <= self.depth_limit:
            if 'text/html' in content_type:
                self.page_count += 1
                text_content = response.xpath('//text()[not(ancestor::style) and not(ancestor::script)]').getall()
                cleaned_text = ' '.join([text.strip() for text in text_content if text.strip()])
                extracted_links = self.extract_links_from_text(cleaned_text, response.url)

                for link in extracted_links:
                    sublink_url = urljoin(response.url, link)
                    sublink_url = urldefrag(sublink_url)[0]
                    if self.is_valid_url(sublink_url) and not self.is_non_text_url(sublink_url) and sublink_url not in self.visited_urls:
                        self.visited_urls.add(sublink_url)
                        self.logger.info(f"Constructed sublink from text: {sublink_url}")
                        yield response.follow(sublink_url, callback=self.parse, meta={'depth': depth + 1})

            else:
                cleaned_text = ''

            item = {'url': response.url, 'parent_url': parent_url, 'content_type': content_type, 'text': cleaned_text}
            self.scraped_items.append(item)

            for link in response.xpath('//a/@href').extract():
                sublink_url = urljoin(response.url, link)
                sublink_url = urldefrag(sublink_url)[0]
                if self.is_valid_url(sublink_url) and not self.is_non_text_url(sublink_url) and sublink_url not in self.visited_urls:
                    self.visited_urls.add(sublink_url)
                    self.logger.info(f"Constructed sublink: {sublink_url}")
                    yield response.follow(sublink_url, callback=self.parse, meta={'depth': depth + 1})

    def closed(self, reason):
        """Handles the closing of the spider and saves scraped items to a JSON file.

        Args:
            reason (str): The reason for closing the spider.
        """
        with open('scraped_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.scraped_items, f, ensure_ascii=False)

    def extract_links_from_text(self, text, base_url):
        """Extracts links embedded within text content.

        Args:
            text (str): The text content to extract links from.
            base_url (str): The base URL for resolving relative links.

        Returns:
            list: A list of extracted links.
        """
        selector = Selector(text=text)
        links = []
        for anchor in selector.xpath('//a'):
            href = anchor.xpath('./@href').get()
            if href:
                links.append(href.strip())
        return links

    def is_non_text_url(self, url):
        """Check if the URL is a non-text URL based on its file extension.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL is a non-text URL, False otherwise.
        """
        parsed_url = urlparse(url)
        file_extension = f'.{parsed_url.path.split(".")[-1].lower()}'
        return file_extension in self.non_text_file_types

    def is_valid_url(self, url):
        """Check if the URL is valid and starts with http or https.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        return url.startswith('http://') or url.startswith('https://')


def scrape_url(start_url, depth, result_queue):
    """Sets up the CrawlerProcess and starts the scraping of the given URL.

    Args:
        start_url (str): The starting URL for scraping.
        depth (int): The maximum depth of the crawl.
        result_queue (multiprocessing.Queue): The queue to store results.
    """
    process = CrawlerProcess(settings={
        'LOG_ENABLED': True,
        'CONCURRENT_REQUESTS': 1024,
        'DOWNLOAD_DELAY': 0,
        'REACTOR_THREADPOOL_MAXSIZE': 200,
        'COOKIES_ENABLED': False,
        'RETRY_ENABLED': False,
        'REDIRECT_ENABLED': False,
    })
    process.crawl(MySpider, start_url=start_url, domain=start_url.split('/')[2], depth=depth, result_queue=result_queue)
    process.start()
    

def run_spider_multiprocessing(start_url, depth, result_queue):
    """Runs the spider in a separate multiprocessing context.

    Args:
        start_url (str): The starting URL for scraping.
        depth (int): The maximum depth of the crawl.
        result_queue (multiprocessing.Queue): The queue to store results.
    """
    scrape_process = Process(target=scrape_url, args=(start_url, depth, result_queue))
    scrape_process.start()
    scrape_process.join()
