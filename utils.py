import re
import os
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
CONTEXT_LENGTH = 30720
import chromadb
from chromadb import Documents, Embeddings, EmbeddingFunction

from typing import List
import google.generativeai as genai
from pypdf import PdfReader

def load_pdf(file_path: str):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    return text


# print(pdf_text)

def split_text(text: str):
    """
    Returns a non empty list of strings given a text string as input
    Splits by "\n \n" to split by a paragraph.

    Parameters:
    """
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i!=""]




class GeminiEmebeddingFunction(EmbeddingFunction):

    def __call__(self, input: Documents) -> Embeddings:
        global gemini_api_key
        if not gemini_api_key:
            raise ValueError("Please provide correct GEMINI_API_KEY.")
        genai.configure(api_key = gemini_api_key)

        genai.configure(api_key = gemini_api_key)
        model = 'models/embedding-001'
        title = "Custom Query"
        
        embedding = genai.embed_content(model=model, 
                                   content = input,     
                                   task_type='retrieval_document',
                                   title=title)['embedding']
        return embedding
    

def create_chroma_db(documents: List, path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmebeddingFunction())

    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))

    return db, name


def load_chroma_collection(path, name):

    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmebeddingFunction())

    return db

def get_relevant_passage(query, db, n_results):
    passage = db.query(query_texts=[query], n_results = n_results)['documents'][0]

    return passage

def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace("\n", "")

    prompt = ('''You are a helpful and informative bot that answers questions using text from the reference passage included below.
    If the passage is irrelevant to the answer, you may ignore it.
    Your sole purpose is to write answers that align with the syllabus given to you. You have to extract information from the book and give relvant content for each topic in the following format. . Don't include any acknowledge anything.
    Topic Name:
    Intuition: <A text that will make it easier to understand the concept by relating it with some real life events or objects>
    Content: <The actual content for academia where precision is absolutely necessary.>
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:


            ''').format(query=query, relevant_passage=escaped)
    
    return prompt
    

def create_answer_template(prompt):
    global gemini_api_key
    if not gemini_api_key:
        raise ValueError("Please provide correct GEMINI_API_KEY.")
    genai.configure(api_key = gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

def generate_answer(db, query):
    relevant_text = get_relevant_passage(query=query, db=db, n_results=3)
    # relevant_text = [x[:CONTEXT_LENGTH*3] for x in relevant_text]
    # Test in exceeded token limit
    print(f"Relevant text : {len(relevant_text[0])} Token: {len(relevant_text[0])/4}")
    prompt = make_rag_prompt(query, relevant_passage="".join(relevant_text))
    answer = create_answer_template(prompt)

    return answer