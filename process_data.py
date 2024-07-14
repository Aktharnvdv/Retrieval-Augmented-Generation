import json
from gensim import corpora, models
import re
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import torch
from gensim.utils import simple_preprocess
import numpy as np
import os

def load_scraped_data(file_path):
    print("--------loading scraped data-------------")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        return scraped_data
    except FileNotFoundError:
        return []

def preprocess_text(text):

    text = re.sub(r'\S*\d\S*', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return [word for word in simple_preprocess(text)]

def chunk_by_topic_modeling(texts):
    print("-------- applying topic modelling on the text -------------")
    
    processed_texts = [preprocess_text(text) for text in texts]
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    lda_model = models.LdaModel(corpus, 
                                num_topics=100, 
                                id2word=dictionary, 
                                passes=100)
    topic_chunks = []
    
    for text in processed_texts:
        bow = dictionary.doc2bow(text)
        topics = lda_model.get_document_topics(bow)
    
        dominant_topic = max(topics, key=lambda x: x[1])[0]
        topic_words = lda_model.show_topic(dominant_topic, topn=1000)
        topic_chunks.append(" ".join([word for word, _ in topic_words]))
    
    return topic_chunks, texts

def save_embeddings(file_path, embeddings, urls, texts):
    print("-------- saving the embeddings -------------")
    
    data = [{"url": url, "embedding": embedding.tolist(), "text": text} for url, embedding, text in zip(urls, embeddings, texts)]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_embeddings(file_path):
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
    scraped_data = load_scraped_data('scraped_data.json')
    if scraped_data:
        texts = [data['text'] for data in scraped_data]
        urls = [data['url'] for data in scraped_data]
        topic_chunks, texts = chunk_by_topic_modeling(texts)
        
        embeddings = create_embeddings(topic_chunks, model, tokenizer)  
        save_embeddings('embeddings.json', embeddings, urls, topic_chunks)
        
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
