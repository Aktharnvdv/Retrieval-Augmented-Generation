**Web Scraping, Data Chunking, and Question Answering System**

*This project involves developing a comprehensive system for web scraping, data chunking, vector database creation, retrieval, re-ranking, and question answering using advanced AI techniques. Below are the detailed tasks and components of the system:*

**Task Description**
*Web Crawling*

    The participant will develop a web crawler to scrape data from the NVIDIA CUDA documentation website:
    Parent Link: NVIDIA CUDA Documentation
    Sub-link Depth: Up to 5 levels (parent link -> sub-link -> sub-link -> sub-link -> sub-link)
    The web crawler should retrieve data from both the parent links and their sub-links.

**Data Chunking and Vector Database Creation**
*Implement sophisticated data chunking techniques based on sentence/topic similarity (e.g., semantic similarity, topic modeling).*
 
    Convert the chunks into embedding vectors.
    Create a vector database using MILVUS.
    Store embedding vectors using FLAT (Flat) and IVF (Inverted File) indexing methods.
    Include metadata such as the web-link of the extracted chunk in the database.

**Retrieval and Re-ranking**

    Employ query expansion techniques to enhance retrieval.
    Use hybrid retrieval methods combining BM25 and BERT/bi-encoder based methods (DPR, Spider) for retrieving relevant data from the vector database.
    Re-rank retrieved data based on relevance and similarity to the query.

**Question Answering**

    Utilize a Language Model (LLM) for question answering.
    Generate accurate answers based on the retrieved and re-ranked data.

**User Interface**
    
    Develop a user interface using frameworks like Streamlit or Gradio.
    Allow users to input queries and display retrieved answers in a user-friendly manner.

*Implementation*

    Prerequisites
    Python 3.7+
    Ensure dependencies are installed (pip install -r requirements.txt)
    Running the System

**Web Crawling and Data Processing:**

    Run the web crawler (python run_spider.py)
    Process and chunk the scraped data (python process_data.py)

**Vector Database Creation:**

    Configure and create MILVUS vector database (pymilvus)

**Retrieval and Question Answering:**

    Implement retrieval and re-ranking strategies (BM25, DPR, Spider)
    Integrate with LLM for question answering (transformers, google-ai-generativelanguage)
    Optional: User Interface:

*Set up Streamlit or Gradio for interactive querying (streamlit run streamlit_app.py)*

**Directory Structure**

    ├── README.md
    ├── requirements.txt
    ├── run_spider.py
    ├── process_data.py
    ├── my_spider.py
    ├── embeddings.json
    ├── streamlit_app.py
    ├── docker-compose.yml
    └── volumes/
        ├── data/
        └── logs/
    
*Notes*

    Replace placeholder URLs (https://docs.nvidia.com/cuda/) with actual links as per project requirements.
    Ensure proper configuration of MILVUS and Docker for database and container management.
