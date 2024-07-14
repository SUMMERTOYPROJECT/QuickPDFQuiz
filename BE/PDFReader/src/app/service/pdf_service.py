import os
from fastapi import UploadFile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

class PDFService:
    def __init__(self):
        self.UPLOAD_DIRECTORY = "temp"
        os.makedirs(self.UPLOAD_DIRECTORY, exist_ok=True)

    async def process_pdf(self, file: UploadFile):
        file_location = await self._save_file(file)
        chunks = self._extract_text(file_location)
        os.remove(file_location)  # 처리 후 임시 파일 삭제
        return {"num_chunks": len(chunks)}

    async def _save_file(self, file: UploadFile):
        file_name = f"{uuid.uuid4()}_{file.filename}"
        file_location = os.path.join(self.UPLOAD_DIRECTORY, file_name)
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        return file_location

    def _extract_text(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(pages)