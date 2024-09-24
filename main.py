from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import json
import shutil
from typing import List, Dict, Any
import ollama

app = FastAPI()

class Question(BaseModel):
    questions: List[str]

def read_uploaded_file(file: UploadFile) -> Dict[str, Any]:
    try:
        if file.content_type == "application/json":
            content = json.loads(file.file.read())
            return content
        elif file.content_type == "application/pdf":
            return {"content": "PDF processing is not implemented yet."}
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.post("/upload/")
async def upload_files(
    questions_file: UploadFile = File(...),  # The questions file (JSON)
    document_file: UploadFile = File(...)    # The document file (JSON or PDF)
):
    try:
        questions_content = read_uploaded_file(questions_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid questions file: {e}")
    
    try:
        document_content = read_uploaded_file(document_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid document file: {e}")

    results = []

    for question in questions_content['questions']:
        context = json.dumps(document_content)  # Convert the document to a JSON string for context
        answer = await get_answer_from_llama(question, context)  # Get the answer from llama
        results.append({"question": question, "answer": answer})

    return {"results": results}

async def get_answer_from_llama(question, context):
    model = "llama2"  # Example: use llama's Llama-2 model
    full_prompt = f"Answer the following question based on this context:\n\nContext: {context}\n\nQuestion: {question}"
    response = ollama.chat(model=model,messages=full_prompt)
    return response["text"]
