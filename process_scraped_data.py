import json

def load_scraped_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def list_all_urls(scraped_data):
    urls = []
    for item in scraped_data:
        urls.append(item['url'])
    return urls

# Load scraped data from JSON file
filename = 'scraped_data.json'
scraped_data = load_scraped_data(filename)

# List all URLs extracted from scraped data
urls = list_all_urls(scraped_data)
print("List of URLs:")
for url in urls:
    print(url)
