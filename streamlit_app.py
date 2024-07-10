import streamlit as st
import json
from urllib.parse import urlparse
import nltk
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility
from multiprocessing import Process, Queue
import google.generativeai as genai
from my_spider import run_spider_multiprocessing
import process_data

# Initialize nltk wordnet
nltk.download('wordnet')

# Setup generative AI model (llm_model)
GOOGLE_API_KEY = 'AIzaSyA6qVsNW-xyvf5ubUz0Ic02I-wsLiM1KHc'
genai.configure(api_key=GOOGLE_API_KEY)

llm_model = genai.GenerativeModel(
    model_name='gemini-1.5-flash-latest',
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
    ]
)

# Function to load previously scraped data
def load_scraped_data():
    try:
        with open('scraped_data.json', 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        return scraped_data
    except FileNotFoundError:
        return []

# Function for query expansion using nltk wordnet
def query_expansion(query):
    synonyms = set()
    for syn in wordnet.synsets(query):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Function for BERT-based document retrieval from Milvus
def bert_based_retrieval(collection, query):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
    query_embedding = model.encode(query)
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "embedding", param=search_params, limit=100)
    retrieved_doc_ids = results[0].ids
    return retrieved_doc_ids

# Function to retrieve texts from Milvus based on document IDs
def retrieve_texts_from_milvus(collection, doc_ids):
    texts = []
    try:
        batch_size = 100
        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i + batch_size]
            expr = f'id in {batch_ids}'
            entities = collection.query(expr, output_fields=["text"])
            for entity in entities:
                texts.append(entity['text'])
    except Exception as e:
        st.error(f"Error retrieving texts from Milvus: {e}")
    return texts

# Main Streamlit application function
def main():
    st.title("Retrieval-Augmented Generation")

    url = st.text_input("Enter a URL to scrape:", "https://docs.nvidia.com/cuda/")
    depth = st.slider("Depth to scrape:", 1, 5, 3)

    result_queue = Queue()

    if st.button("Scrape"):
        st.info(f"Scraping {url} up to depth {depth}...")
        result_queue = Queue()
        scrape_process = Process(target=run_spider_multiprocessing, args=(url, depth, result_queue))
        scrape_process.start()
        
        with st.spinner('Scraping in progress...'):
            scrape_process.join()
        
        scraped_data = load_scraped_data()
        
        if scraped_data:
            process_data.main()
            
            st.success("Data processed and stored in Milvus successfully.")
        else:
            st.warning("No data scraped.")

    st.subheader("Retrieve Documents")
    query = st.text_input("Enter your query:")
    if st.button("Retrieve"):
        if query:
            connections.connect("default", host="localhost", port="19530")
            if not utility.has_collection("topic_chunks"):
                st.warning("Collection 'topic_chunks' does not exist")
                return

            collection = Collection("topic_chunks")
            collection.load()

            expanded_query_terms = query_expansion(query)
            expanded_query = " ".join(expanded_query_terms)

            retrieved_doc_ids = bert_based_retrieval(collection, expanded_query)
            retrieved_texts = retrieve_texts_from_milvus(collection, retrieved_doc_ids)

            if not retrieved_texts:
                st.warning("No texts retrieved from Milvus.")
                return

            try:
                documents = [{"text": text} for text in retrieved_texts[:20]]
                prompt = f"For query: '{query}', refer to the following documents: {documents}"
                llm_response = llm_model.generate_content(prompt)
                st.subheader("LLM Response:")
                st.write(llm_response.text)
            except Exception as e:
                st.error(f"Error generating response from LLM: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
