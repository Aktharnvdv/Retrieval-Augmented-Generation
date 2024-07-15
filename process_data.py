import json
from gensim import corpora, models
import re
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import torch
from gensim.utils import simple_preprocess
import numpy as np
import os
from sklearn.cluster import KMeans

def load_scraped_data(file_path):
    """Loads scraped data from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing scraped data.

    Returns:
        list: A list of scraped data items.
    """
    print("--------loading scraped data-------------")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        return scraped_data
    except FileNotFoundError:
        return []

def preprocess_text(text):
    """Cleans and preprocesses the input text.

    Args:
        text (str): The text to preprocess.

    Returns:
        list: A list of cleaned and tokenized words.
    """
    text = re.sub(r'\S*\d\S*', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return [word for word in simple_preprocess(text)]

def chunk_by_similarity(sentences, model, tokenizer, n_clusters=10):
    """Chunks text into clusters based on semantic similarity using a BERT model.

    Args:
        sentences (list): A list of sentences to chunk.
        model (torch.nn.Module): The model used to generate embeddings.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        n_clusters (int): The number of clusters to create.

    Returns:
        dict: A dictionary where keys are cluster labels and values are lists of sentences in each cluster.
    """
    def get_embeddings(sentences):
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    embeddings = get_embeddings(sentences)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(sentences[idx])
    return clusters

def save_embeddings(file_path, embeddings, urls, texts):
    """Saves embeddings, URLs, and texts to a JSON file.

    Args:
        file_path (str): The path to the JSON file to save data.
        embeddings (np.ndarray): The embeddings to save.
        urls (list): The corresponding URLs.
        texts (list): The corresponding texts.
    """
    print("-------- saving the embeddings -------------")
    
    data = [{"url": url, "embedding": embedding.tolist(), "text": text} for url, embedding, text in zip(urls, embeddings, texts)]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_embeddings(file_path):
    """Loads embeddings, URLs, and texts from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing embeddings.

    Returns:
        tuple: A tuple containing:
            - list: Loaded embeddings.
            - list: Corresponding URLs.
            - list: Corresponding texts.
    """
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
    """Connects to the Milvus database and creates a collection if it does not exist.

    Returns:
        pymilvus.Collection: The Milvus collection object.
    """
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
    """Inserts embeddings, URLs, and texts into the specified Milvus collection.

    Args:
        collection (pymilvus.Collection): The Milvus collection to insert data into.
        embeddings (list): The embeddings to insert.
        urls (list): The corresponding URLs.
        texts (list): The corresponding texts.
    """
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

def create_embeddings(texts, model, tokenizer, batch_size=8):
    """Creates embeddings for the provided texts using the specified model and tokenizer.

    Args:
        texts (list): List of texts to create embeddings for.
        model (torch.nn.Module): The model used to generate embeddings.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        batch_size (int, optional): The number of texts to process in each batch. Default is 8.

    Returns:
        np.ndarray: The combined embeddings for all texts.
    """
    print("-------- creating embeddings -------------")
    
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

def main(model, tokenizer):
    """Main function to process scraped data, create embeddings, and insert them into Milvus.

    Args:
        model (torch.nn.Module): The model used for embedding creation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
    """
    scraped_data = load_scraped_data('scraped_data.json')
    if scraped_data:
        texts = [data['text'] for data in scraped_data]
        urls = [data['url'] for data in scraped_data]
        
        # Use chunk_by_similarity instead of chunk_by_topic_modeling
        clusters = chunk_by_similarity(texts, model, tokenizer, n_clusters=10)
        
        # Flatten clusters into a list of texts
        chunked_texts = []
        for cluster in clusters.values():
            chunked_texts.extend(cluster)
        
        embeddings = create_embeddings(chunked_texts, model, tokenizer)  
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
