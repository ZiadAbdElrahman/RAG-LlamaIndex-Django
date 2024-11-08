from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

from rag import RAG
from config import config
from custom_logger import get_logger

logger = get_logger()
rag = RAG()
app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class get_respond_request(BaseModel):
    user_id: str
    message: str
    
class uplaod_request(BaseModel):
    user_id: str
    file: UploadFile = File(...)
    

@app.post("/get_response/")
async def get_response(req_input: get_respond_request):
    logger.info(f'received a message from {req_input.user_id}')
    response = process_user_input(req_input.user_id, req_input.message)
    
    return {"input": req_input.message, "response": response}

def process_user_input(user_id, input_text):
    output = rag.get_respond(user_id, input_text)
    return output

@app.post("/upload_pdf/")
async def upload_pdf(user_id: str = Form(...), file: UploadFile = File(...)):
    logger.info('received pdf upload request')
    
    if file.content_type != 'application/pdf':
        return {"message": "Invalid file type, expected PDF"}
    try:
        file_location = f"./uploaded_files/{user_id}_{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
        
    rag.add_pdf_file(user_id, file_location)
    return {"message": "File uploaded successfully.", "filename": file.filename}
