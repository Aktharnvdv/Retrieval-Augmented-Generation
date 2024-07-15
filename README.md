# Web Scraping, Data Chunking, and Question Answering System

*This project involves developing a comprehensive system for web scraping, data chunking, vector database creation, retrieval, re-ranking, and question answering using advanced AI techniques. Below are the detailed tasks and components of the system:*

## Directory Structure
    
    ├── volumes/
    │   ├── data/
    │   └── logs/
    ├── docker-compose.yml
    ├── my_spider.py
    ├── process_data.py
    ├── README.md
    ├── requirements.txt
    ├── setup.sh
    ├── streamlit_app.py

## Set up the repository

    git clone https://github.com/Aktharnvdv/Retrieval-Augmented-Generation.git
    cd Retrieval-Augmented-Generation
    ./setup.sh

## Task Description

### Web Crawling
     
Web crawler was developed using Scrapy, which by default scrapes the Nvidia documentation website. It extracts text from the main page as well as sublinks, scraping data up to a depth of 5.web crawler (my_spider.py)

### Data Chunking and Vector Database Creation

Milvus is the vector database used and topic modeling is implemented to chunk the scraped data using the Gensim library.
 
- Converted the texts into chunks using topic modeling.
- Chunks were converted into embedding vectors using BERT models.
- Created a vector database using Milvus.
- Stored embedding vectors using FLAT (Flat) and IVF (Inverted File) indexing methods.
- Included metadata such as the web-link of the extracted chunk, embeddings, and texts in the database.

    Process and chunk the scraped data (process_data.py)
    Configure and create Milvus vector database (pymilvus)
    sudo docker-compose up -d

### Retrieval and Re-ranking

- Employed query expansion techniques to enhance retrieval.
- Used hybrid retrieval methods combining BM25 and BERT/bi-encoder based method DPR for retrieving relevant data from the vector database.
- Re-ranked retrieved data based on relevance and similarity to the query.
- Implemented retrieval and re-ranking strategies (BM25, DPR)

### Question Answering

- Utilized a language model (Google Gemini) for question answering.
- Generated accurate answers based on the retrieved and re-ranked data.
- Integrated with LLM for question answering (transformers, google-ai-generativelanguage (Gemini))

### User Interface
    
Developed a user interface using frameworks like Streamlit, allowing users to input queries and display retrieved answers in a user-friendly manner.
