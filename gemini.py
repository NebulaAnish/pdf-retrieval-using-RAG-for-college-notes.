import os
from utils import *

pdf_text = load_pdf("introduction.pdf")
chunked_text = split_text(text=pdf_text)
# print(chunked_text[0])


# # One time process
try:
    db, name = create_chroma_db(documents=chunked_text,
                                path='contents/',
                                name='collection_name')
except:
    pass

db = load_chroma_collection(path='contents/', name='collection_name')
answer = generate_answer(db=db, query='Mesh Topology')


print(answer)