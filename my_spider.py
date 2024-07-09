import scrapy
import json
from urllib.parse import urljoin, urlparse, urldefrag
from scrapy.crawler import CrawlerProcess
from scrapy.selector import Selector

class MySpider(scrapy.Spider):
    name = 'MySpider'

    def __init__(self, start_url=None, domain=None, depth=1, *args, **kwargs):
        super(MySpider, self).__init__(*args, **kwargs)
        self.start_url = start_url
        self.allowed_domains = [domain]
        self.depth_limit = int(depth)
        self.scraped_items = []
        self.non_text_file_types = ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.doc', '.docx', '.xls', '.xlsx']
        self.page_count = 0
        self.max_pages = 100  

    def start_requests(self):
        yield scrapy.Request(url=self.start_url, callback=self.parse, meta={'depth': 0})

    def parse(self, response):
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
                    if not self.is_non_text_url(sublink_url):
                        self.logger.info(f"Constructed sublink from text: {sublink_url}")
                        yield response.follow(sublink_url, callback=self.parse, meta={'depth': depth + 1})

            else:
                cleaned_text = ''

            item = {'url': response.url, 'parent_url': parent_url, 'content_type': content_type, 'text': cleaned_text}
            self.scraped_items.append(item)

            for link in response.xpath('//a/@href').extract():
                sublink_url = urljoin(response.url, link)
                sublink_url = urldefrag(sublink_url)[0]  
                if not self.is_non_text_url(sublink_url):
                    self.logger.info(f"Constructed sublink: {sublink_url}")
                    yield response.follow(sublink_url, callback=self.parse, meta={'depth': depth + 1})

    def closed(self, reason):
        with open('scraped_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.scraped_items, f, ensure_ascii=False)

    def extract_links_from_text(self, text, base_url):
        """Extracts links embedded within text content."""
        selector = Selector(text=text)
        links = []
        for anchor in selector.xpath('//a'):
            href = anchor.xpath('./@href').get()
            if href:
                links.append(href.strip())
        return links

    def is_non_text_url(self, url):
        """Check if the URL is a non-text URL based on its file extension."""
        parsed_url = urlparse(url)
        file_extension = parsed_url.path.split('.')[-1].lower()
        return any(url.endswith(ext) for ext in self.non_text_file_types)

#def main():
#    start_url = "https://docs.nvidia.com/cuda/"
#    domain = "docs.nvidia.com"
#    depth = 4
#    process = CrawlerProcess(settings={'LOG_ENABLED': True,})
#    process.crawl(MySpider, start_url=start_url, domain=domain, depth=depth)
#    process.start()

#if __name__ == "__main__":
#    main()

