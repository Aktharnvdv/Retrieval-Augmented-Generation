import streamlit as st
import json
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from pymilvus import connections, Collection, utility
from multiprocessing import Process, Queue
import google.generativeai as genai
from my_spider import run_spider_multiprocessing
import process_data
from transformers import AutoTokenizer, AutoModel
import bm25s
import torch
import os

# Ensure NLTK data is downloaded
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure Google API key and LLM model
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

@st.cache_resource
def load_bert_model():
    """Loads and returns the BERT tokenizer and model.

    Returns:
        tokenizer (transformers.AutoTokenizer): The BERT tokenizer.
        model (transformers.AutoModel): The BERT model.
    """
    print("-------- loading bert model -------------")
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.eval()  
    return tokenizer, model

tokenizer, model = load_bert_model()

def load_scraped_data():
    """Loads previously scraped data from 'scraped_data.json'.

    Returns:
        list: The list of scraped data items.
    """
    print("-------- loading the scraped data -------------")
    
    try:
        with open('scraped_data.json', 'r', encoding='utf-8') as f:
            scraped_data = json.load(f)
        return scraped_data
    except FileNotFoundError:
        return []

def query_expansion(query):
    """Expands the query using synonyms from WordNet.

    Args:
        query (str): The query string to expand.

    Returns:
        list: List of expanded query terms.
    """
    print("-------- query expansion -------------")
    
    synonyms = set()
    for syn in wordnet.synsets(query):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def bert_based_retrieval(collection, query, tokenizer, model):
    """Executes BERT-based retrieval on a Milvus collection.

    Args:
        collection (pymilvus.Collection): The Milvus collection object.
        query (str): The query string.
        tokenizer (transformers.AutoTokenizer): The BERT tokenizer.
        model (transformers.AutoModel): The BERT model.

    Returns:
        list: List of retrieved texts.
    """
    print("-------- executing DRP bert method -------------")
    
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search([query_embedding[0]], "embedding", param=search_params, limit=100, output_fields=["text"])
    retrieved_texts = [result.entity.get("text") for result in results[0]]
    return retrieved_texts

def bm25_re_rank(retrieved_texts, query):
    """Applies BM25 re-ranking on retrieved texts.

    Args:
        retrieved_texts (list): List of texts retrieved from BERT.
        query (str): The query string.

    Returns:
        list: List of ranked texts.
    """
    print("-------- applying bm25 on the retrieved chunks -------------")
    
    stop_words = set(stopwords.words('english'))
    corpus_tokens = [word_tokenize(doc.lower()) for doc in retrieved_texts]
    corpus_tokens = [[word for word in tokens if word.isalnum() and word not in stop_words] for tokens in corpus_tokens]
    
    query_tokens = word_tokenize(query.lower())
    query_tokens = [word for word in query_tokens if word.isalnum() and word not in stop_words]
    
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    results, scores = retriever.retrieve(query_tokens, k=len(retrieved_texts))
    
    ranked_texts = [retrieved_texts[idx] for idx in results[0]]
    return ranked_texts

def main():
    """Main function to run the Streamlit application for Retrieval-Augmented Generation."""
    st.title("Retrieval-Augmented Generation")
    print("-------- Retrieval-Augmented Generation -------------------")

    url = st.text_input("Enter a URL to scrape:", "https://docs.nvidia.com/cuda/")
    depth = 5
    result_queue = Queue()

    if st.button("Scrape"):
        st.info(f"Scraping {url} up to depth {depth}...")
        result_queue = Queue()
        scrape_process = Process(target=run_spider_multiprocessing,args=(url, depth, result_queue))
        scrape_process.start()
        
        with st.spinner('Scraping in progress...'):
            scrape_process.join()
        
        scraped_data = load_scraped_data()
        
        if scraped_data:
            process_data.main(model , tokenizer)
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
            retrieved_texts = bert_based_retrieval(collection, expanded_query, tokenizer, model)
            
            if not retrieved_texts:
                st.warning("No texts retrieved from BERT.")
                return

            ranked_texts = bm25_re_rank(retrieved_texts, query)

            if not ranked_texts:
                st.warning("No texts retrieved after BM25 re-ranking.")
                return

            try:
                documents = [{"text": text} for text in ranked_texts[:20]]
                prompt = f"For query: '{query}', refer to the following documents: {documents} answer the question with citation from given reference"
                llm_response = llm_model.generate_content(prompt)
                st.subheader("LLM Response:")
                st.write(llm_response.text)
                print("result published:")
            except Exception as e:
                st.error(f"Error generating response from LLM: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
