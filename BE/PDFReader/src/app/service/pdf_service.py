import os
import uuid
import json
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import asyncio
import logging
from datetime import datetime, timedelta
import aiofiles
import hashlib
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        self.UPLOAD_DIRECTORY = "temp"
        self.STATUS_FILE = "status_info.json"
        self.VECTOR_STORE_PATH = "faiss_index"
        os.makedirs(self.UPLOAD_DIRECTORY, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = self.load_vector_store()
        self.processing_status = {}
        self.status_expiry = {}
        self.load_status_info()
        self.generated_questions = set()

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

    def load_vector_store(self):
        try:
            if os.path.exists(self.VECTOR_STORE_PATH):
                logger.info(f"Loading existing vector store from {self.VECTOR_STORE_PATH}")
                return FAISS.load_local(self.VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization=True)
            else:
                logger.info("Creating new vector store")
                return FAISS.from_texts(["initialization"], self.embeddings)
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return FAISS.from_texts(["initialization"], self.embeddings)

    def save_vector_store(self):
        logger.info(f"Saving vector store to {self.VECTOR_STORE_PATH}")
        self.vector_store.save_local(self.VECTOR_STORE_PATH)

    def load_status_info(self):
        if os.path.exists(self.STATUS_FILE):
            with open(self.STATUS_FILE, 'r') as f:
                data = json.load(f)
                self.processing_status = data['processing_status']
                self.status_expiry = {k: datetime.fromisoformat(v) for k, v in data['status_expiry'].items()}
        logger.info(f"Loaded status info. Processing status count: {len(self.processing_status)}")

    def save_status_info(self):
        with open(self.STATUS_FILE, 'w') as f:
            json.dump({
                'processing_status': self.processing_status,
                'status_expiry': {k: v.isoformat() for k, v in self.status_expiry.items()}
            }, f)
        logger.info(f"Saved status info. Processing status count: {len(self.processing_status)}")

    async def save_pdf(self, file: UploadFile):
        try:
            file_id = str(uuid.uuid4())
            file_name = f"{file_id}_{file.filename}"
            file_location = os.path.join(self.UPLOAD_DIRECTORY, file_name)

            logger.info(f"Attempting to save file: {file_name}")
            async with aiofiles.open(file_location, "wb") as out_file:
                content = await file.read()
                await out_file.write(content)

            logger.info(f"PDF saved successfully: {file_id} at location: {file_location}")

            self.processing_status[file_id] = "Uploaded"
            self.update_status_expiry(file_id)
            self.save_status_info()
            logger.info(f"Status updated for file_id: {file_id}. Status: Uploaded")

            logger.info(f"Starting background processing for file_id: {file_id}")
            asyncio.create_task(self.process_pdf(file_location, file_id))

            return {"file_id": file_id, "file_location": file_location}
        except Exception as e:
            logger.error(f"Error saving PDF: {str(e)}")
            raise

    async def process_pdf(self, file_location: str, file_id: str):
        try:
            self.processing_status[file_id] = "Processing"
            self.update_status_expiry(file_id)
            self.save_status_info()
            logger.info(f"Starting PDF processing: {file_id}")

            chunks = await self._extract_text(file_location)
            await self._add_to_vector_store(chunks, file_id)

            self.processing_status[file_id] = "Completed"
            self.update_status_expiry(file_id, hours=168)  # 7 days
            self.save_status_info()
            logger.info(f"PDF processing completed: {file_id}")
            self.save_vector_store()
        except Exception as e:
            logger.error(f"Error processing PDF {file_id}: {str(e)}")
            self.processing_status[file_id] = f"Error: {str(e)}"
            self.update_status_expiry(file_id)
            self.save_status_info()
        finally:
            if os.path.exists(file_location):
                os.remove(file_location)
                logger.info(f"Temporary file removed: {file_location}")
            else:
                logger.warning(f"Temporary file not found for removal: {file_location}")

    def update_status_expiry(self, file_id: str, hours: int = 24):
        self.status_expiry[file_id] = datetime.now() + timedelta(hours=hours)
        logger.info(f"Updated expiry for file_id: {file_id}. New expiry: {self.status_expiry[file_id]}")

    async def get_processing_status(self, file_id: str):
        self.clean_expired_statuses()
        status = self.processing_status.get(file_id)
        expiry = self.status_expiry.get(file_id)

        if status is None:
            if file_id in self.status_expiry:
                logger.warning(f"Status expired for file_id: {file_id}")
                return {"status": "Expired", "message": "Status information has expired"}
            else:
                logger.warning(f"No status found for file_id: {file_id}")
                logger.info(f"Current processing_status keys: {list(self.processing_status.keys())}")
                logger.info(f"Current status_expiry keys: {list(self.status_expiry.keys())}")
                return {"status": "Not Found", "message": "No status information found for this file ID"}

        logger.info(f"Status for {file_id}: {status}")
        self.update_status_expiry(file_id)
        self.save_status_info()

        return {
            "status": status,
            "expiry": expiry.isoformat() if expiry else None
        }

    def clean_expired_statuses(self):
        now = datetime.now()
        expired_ids = [file_id for file_id, expiry_time in self.status_expiry.items() if expiry_time < now]
        for file_id in expired_ids:
            self.processing_status.pop(file_id, None)
            self.status_expiry.pop(file_id, None)
            logger.info(f"Expired status removed for file_id: {file_id}")
        if expired_ids:
            self.save_status_info()

    async def _extract_text(self, file_path: str):
        logger.info(f"Extracting text from PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(pages)

    async def _add_to_vector_store(self, chunks, file_id: str):
        logger.info(f"Adding {len(chunks)} chunks to vector store for file_id: {file_id}")
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [{"file_id": file_id, "chunk_id": i} for i in range(len(chunks))]
        try:
            self.vector_store.add_texts(texts, metadatas)
            logger.info(f"Successfully added {len(chunks)} chunks to vector store for file_id: {file_id}")
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {str(e)}")

    async def search_vectors(self, query: str, file_id: str = None, k: int = 5):
        logger.info(f"Searching vectors with query: '{query}', file_id: {file_id}")
        try:
            if file_id:
                filter_dict = {"file_id": file_id}
                results = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
            else:
                results = self.vector_store.similarity_search(query, k=k)

            logger.info(f"Search completed. Number of results: {len(results)}")
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            logger.error(f"Error in search_vectors: {str(e)}")
            return []

    def _get_question_hash(self, question):
        return hashlib.md5(question.encode()).hexdigest()

    async def generate_single_question(self, content: str, difficulty: str, previous_questions: list):
        prompt = PromptTemplate(
            input_variables=["content", "difficulty", "previous_questions"],
            template="""
            다음 내용을 바탕으로 하나의 객관식 퀴즈 문제를 만들어주세요.

            문제는 다음 요소를 포함해야 합니다:
            1. 난이도 표시 ({difficulty})
            2. 명확하고 간결한 질문
            3. 4개의 선택지 (a, b, c, d)
            4. 정확한 정답 표시
            5. 간단한 해설

            난이도에 따른 문제 특성:
            - 쉬움: 직접적인 사실이나 정보를 묻는 질문
            - 중간: 약간의 추론이나 연결이 필요한 질문
            - 어려움: 깊은 이해나 여러 개념의 연결이 필요한 질문
            - 가장 어려움: 고차원적 사고나 복잡한 분석이 필요한 질문

            내용:
            {content}

            이전에 생성된 질문들:
            {previous_questions}

            퀴즈 형식:
            [난이도: {difficulty}]
            [질문]
            a) [선택지1]
            b) [선택지2]
            c) [선택지3]
            d) [선택지4]
            정답: [정답 선택지]
            해설: [문제에 대한 설명 및 정답 근거]

            위의 형식에 따라 정확히 하나의 퀴즈 문제를 생성해주세요. 이전에 생성된 질문들과 중복되지 않도록 주의하고, 내용의 다른 부분에 대해 질문하세요:
            """
        )

        quiz_chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            result = await asyncio.to_thread(
                quiz_chain.invoke,
                {
                    "content": content,
                    "difficulty": difficulty,
                    "previous_questions": "\n".join(previous_questions)
                }
            )
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            return None

    async def generate_quiz(self, file_id: str, num_questions: int = 10):
        logger.info(f"Generating quiz for file_id: {file_id}")

        results = await self.search_vectors("", file_id=file_id, k=10)

        if not results:
            logger.warning(f"No content found for file_id: {file_id}")
            return {"error": "No content found for the given file ID"}

        content = " ".join([result['content'] for result in results])
        logger.info(f"Combined content length: {len(content)} characters")

        difficulty_distribution = {
            "쉬움": int(num_questions * 0.4),
            "중간": int(num_questions * 0.3),
            "어려움": int(num_questions * 0.2),
            "가장 어려움": num_questions - int(num_questions * 0.4) - int(num_questions * 0.3) - int(num_questions * 0.2)
        }

        quiz_questions = []
        previous_questions = []

        for difficulty, count in difficulty_distribution.items():
            attempts = 0
            while len([q for q in quiz_questions if difficulty in q]) < count and attempts < count * 2:
                question = await self.generate_single_question(content, difficulty, previous_questions)
                if question and self._get_question_hash(question) not in self.generated_questions:
                    quiz_questions.append(question)
                    self.generated_questions.add(self._get_question_hash(question))
                    previous_questions.append(question.split('\n')[0])  # 질문만 추가
                attempts += 1

        # 번호 매기기
        numbered_questions = []
        for i, question in enumerate(quiz_questions, 1):
            lines = question.split('\n')
            lines[0] = f"{i}. {lines[0]}"
            numbered_questions.append('\n'.join(lines))

        logger.info(f"Generated {len(numbered_questions)} questions for file_id: {file_id}")
        return {"quiz": numbered_questions}

    async def get_quiz_questions(self, file_id: str, num_questions: int = 10):
        results = await self.search_vectors("", file_id=file_id, k=10)

        if not results:
            yield {"error": "No content found for the given file ID"}
            return

        content = " ".join([result['content'] for result in results])

        difficulty_distribution = {
            "쉬움": int(num_questions * 0.4),
            "중간": int(num_questions * 0.3),
            "어려움": int(num_questions * 0.2),
            "가장 어려움": num_questions - int(num_questions * 0.4) - int(num_questions * 0.3) - int(num_questions * 0.2)
        }

        previous_questions = []
        question_number = 1

        for difficulty, count in difficulty_distribution.items():
            for _ in range(count):
                question = await self.generate_single_question(content, difficulty, previous_questions)
                if question and self._get_question_hash(question) not in self.generated_questions:
                    lines = question.split('\n')
                    lines[0] = f"{question_number}. {lines[0]}"
                    numbered_question = '\n'.join(lines)

                    self.generated_questions.add(self._get_question_hash(question))
                    previous_questions.append(question.split('\n')[0])  # 질문만 추가
                    question_number += 1

                    yield {"quiz": numbered_question}
                else:
                    yield {"error": f"Failed to generate unique question for difficulty: {difficulty}"}