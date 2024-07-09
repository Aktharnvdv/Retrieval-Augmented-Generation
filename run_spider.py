from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from my_spider import MySpider
from multiprocessing import Process

def scrape_url(start_url, depth, result_queue):
    process = CrawlerProcess(get_project_settings())
    process.crawl(MySpider, start_url=start_url, domain=start_url.split('/')[2], depth=depth, result_queue=result_queue)
    process.start()

def run_spider_multiprocessing(start_url, depth, result_queue):
    scrape_process = Process(target=scrape_url, args=(start_url, depth, result_queue))
    scrape_process.start()
    scrape_process.join()
