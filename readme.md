# RAG using gemini model
The following is a test implementation of RAG by using a PDF for document retrieval with Gemini API. It used chromaDB as the vector database to store embeddings of the uploaded pdf. 
It doesn't cover the edges cases as of the initial version.

### Steps to test
Create a `.env` file with your API_KEY of google's gemini. 
1. `$pip install -r requirements.txt`
2. `$python gemini.py`

Query can be changed from `gemini.py` file.


## Steps done in the code 
(`utils.py` consists of core logic)
### Step 1: Loading pdf content
    It is done by using `pypdf` library that let's us extract the contents of a pdf as a single string

### Step 2: Split into chunks
    Since LLMs are restricted by their context length, so we'll divide the text into chunks of small size. Each paragraph is split into chunks for the sake of simplicity.

### Step 3: Create embeddings

### Step 4: Store embeddings in a vector db
    Stored in chroma db using a certain collection_name which can be used in future to load this collection.

### Step 5: Load collection
    The collection_name and path of the stored db can be used to load this collection