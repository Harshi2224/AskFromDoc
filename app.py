Hugging Face's logo
Hugging Face
Models
Datasets
Spaces
Community
Docs
Pricing



Hugging Face is way more fun with friends and colleagues! 🤗 Join an organization
Spaces:

HarshithaSHarshi
/
AskFromDoc


like
0

App
Files
Community
Settings
AskFromDoc
/
app.py

HarshithaSHarshi's picture
HarshithaSHarshi
Update app.py
3f6ba4d
verified
3 days ago
raw

Copy download link
history
blame
edit
delete

1.73 kB
import os
from flask import Flask, request, jsonify, send_from_directory
os.environ['HF_HOME'] = '/tmp'
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.schema import Document
from flask_cors import CORS
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
llm = HuggingFacePipeline(pipeline = pipe)
app = Flask(__name__)
CORS(app)
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')
@app.route('/generate', methods=['POST'])
def generate():
    file = request.files['file']
    Question=request.form['Question']
    text = file.read().decode('utf-8')
    text_splitter = CharacterTextSplitter(chunk_size = 300,chunk_overlap=70)
    # Create a Document object from your text
    doc = Document(page_content=text)
    docs = text_splitter.split_documents([doc])
    db = FAISS.from_documents(docs,embedding)
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)
    result = qa.invoke(Question)
    output=[]
    output.append(result)
    return jsonify(output)
if __name__ == '__main__': 
    app.run(host='0.0.0.0',port=7860)
