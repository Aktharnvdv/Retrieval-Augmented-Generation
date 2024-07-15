import json
import re
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import torch
from gensim.utils import simple_preprocess
import numpy as np
from sklearn.cluster import KMeans

def load_scraped_data(file_path):
    """Loads scraped data from a JSON file."""
    print("--------loading scraped data-------------")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        return scraped_data
    except FileNotFoundError:
        return []

def preprocess_text(text):
    """Cleans and preprocesses the input text."""
    text = re.sub(r'\S*\d\S*', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return [word for word in simple_preprocess(text)]

def chunk_by_similarity(sentences, model, tokenizer, n_clusters=10):
    """Chunks text into clusters based on semantic similarity using a BERT model."""
    
    def get_embeddings(sentences, model, tokenizer, batch_size=2):
        """Generates embeddings for the sentences with batch processing."""
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            inputs = tokenizer(batch_sentences, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)

    embeddings = get_embeddings(sentences, model, tokenizer)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(sentences[idx])
    
    return clusters, embeddings  # Return embeddings along with clusters

def save_embeddings(file_path, embeddings, urls, texts):
    """Saves embeddings, URLs, and texts to a JSON file."""
    print("-------- saving the embeddings -------------")
    data = [{"url": url, "embedding": embedding.tolist(), "text": text} for url, embedding, text in zip(urls, embeddings, texts)]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_embeddings(file_path):
    """Loads embeddings, URLs, and texts from a JSON file."""
    print("-------- loading the embeddings -------------")
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
    """Connects to the Milvus database and creates a collection if it does not exist."""
    print("-------- connecting the milvus -------------")
    connections.connect("default", host="localhost", port="19530")
    if utility.has_collection("topic_chunks"):
        utility.drop_collection("topic_chunks")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]

    schema = CollectionSchema(fields, "Topic chunks collection")
    collection = Collection("topic_chunks", schema)
    index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
    collection.create_index("embedding", index_params)

    return collection

def insert_into_milvus(collection, embeddings, urls, texts):
    """Inserts embeddings, URLs, and texts into the specified Milvus collection."""
    print("-------- inserting into milvus -------------")
    entities = {"embedding": embeddings, "url": urls, "text": [text[:65530] for text in texts]}
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

def main(model, tokenizer):
    """Main function to process scraped data, create embeddings, and insert them into Milvus."""
    scraped_data = load_scraped_data('scraped_data.json')
    if scraped_data:
        texts = [data['text'] for data in scraped_data]
        urls = [data['url'] for data in scraped_data]
        
        # Use chunk_by_similarity instead of chunk_by_topic_modeling
        clusters, embeddings = chunk_by_similarity(texts, model, tokenizer, n_clusters=10)
        
        # Flatten clusters into a list of texts
        chunked_texts = []
        for cluster in clusters.values():
            chunked_texts.extend(cluster)
        
        save_embeddings('embeddings.json', embeddings, urls, chunked_texts)
        
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


#if __name__ == "__main__":
#    main()