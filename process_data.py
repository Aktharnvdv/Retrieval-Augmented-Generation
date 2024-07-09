import json
from gensim import corpora, models
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

def load_scraped_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        return scraped_data
    except FileNotFoundError:
        return []

def chunk_by_topic_modeling(texts):
    texts_tokenized = [text.split() for text in texts]
    dictionary = corpora.Dictionary(texts_tokenized)
    corpus = [dictionary.doc2bow(text) for text in texts_tokenized]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
    topic_chunks = []
    for text in texts_tokenized:
        bow = dictionary.doc2bow(text)
        topics = lda_model.get_document_topics(bow)
        topic_chunks.append(" ".join(dictionary[word_id] for word_id, _ in topics))
    return topic_chunks, texts

def save_embeddings(file_path, embeddings, urls, texts):
    data = [{"url": url, "embedding": embedding.tolist(), "text": text} for url, embedding, text in zip(urls, embeddings, texts)]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_embeddings(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        embeddings = [item['embedding'] for item in data]
        urls = [item['url'] for item in data]
        texts = [item['text'] for item in data]
        return embeddings, urls, texts
    except FileNotFoundError:
        return [], [], []

def connect_milvus():
    connections.connect("default", host="localhost", port="19530")
    if utility.has_collection("topic_chunks"):
        utility.drop_collection("topic_chunks")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]

    schema = CollectionSchema(fields, "Topic chunks collection")
    collection = Collection("topic_chunks", schema)
    index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
    collection.create_index("embedding", index_params)

    return collection

def insert_into_milvus(collection, embeddings, urls, texts):
    entities = {
        "embedding": embeddings,
        "url": urls,
        "text": [text[:65530] for text in texts]
    }

    try:
        if entities["embedding"] and entities["url"] and entities["text"]:
            collection.insert([entities["embedding"], entities["url"], entities["text"]])
            collection.flush()
            print("Data inserted into Milvus successfully.")
        else:
            print("No entities to insert into Milvus.")
    except Exception as e:
        print(f"Error inserting data into Milvus: {e}")
        import traceback
        traceback.print_exc()

def main():
    scraped_data = load_scraped_data('scraped_data.json')

    if scraped_data:
        texts = [data['text'] for data in scraped_data]
        urls = [data['url'] for data in scraped_data]
        _ , texts = chunk_by_topic_modeling(texts)
        
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
        embeddings = model.encode(texts)
        save_embeddings('embeddings.json', embeddings, urls, texts)
        
        print("Embeddings saved to embeddings.json successfully.")
        embeddings, urls, texts = load_embeddings('embeddings.json')
        if embeddings and urls:
            collection = connect_milvus()
            print("Connection successful")
            insert_into_milvus(collection, embeddings, urls, texts)
        else:
            print("No embeddings available to insert.")
    else:
        print("No scraped data available.")

if __name__ == "__main__":
    main()
