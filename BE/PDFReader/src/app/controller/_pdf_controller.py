#
# from fastapi import APIRouter, UploadFile, File
# import aiofiles
# from fastapi.responses import JSONResponse
# import logging
# router = APIRouter()
# logger = logging.getLogger(__name__)
# import  os
# @router.get("/check")
# async def health_check():
#     return {"message": "It's Working On PDF Controller Service!"}
#
# import PyPDF2
#
# def extract_text_from_pdf(file_path):
#     with open(file_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text()
#     return text
# @router.post("/upload-pdf")
# async def upload_pdf(file: UploadFile = File(...)):
#     try:
#         # 임시 디렉토리에 파일 저장
#         file_location = f"./src/app/temp/{file.filename}"
#         async with aiofiles.open(file_location, 'wb') as out_file:
#             content = await file.read()
#             await out_file.write(content)
#
#             # 텍스트 추출
#         extracted_text = extract_text_from_pdf(file_location)
#
#         # 퀴즈 생성
#         quiz = generate_quiz(extracted_text)
#             # 임시 파일 삭제
#         os.remove(file_location)
#
#         return {"filename": file.filename, "quiz": quiz}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"message": str(e)})
#
#
# import random
# import nltk
#
# nltk.download('punkt')
#
#
# def generate_quiz(text):
#     sentences = nltk.sent_tokenize(text)
#     quiz_questions = []
#
#     for sentence in sentences[:5]:  # 처음 5개 문장만 사용
#         words = sentence.split()
#         if len(words) > 5:  # 문장이 충분히 길 경우에만
#             blank_index = random.randint(1, len(words) - 2)  # 첫 단어와 마지막 단어는 제외
#             answer = words[blank_index]
#             words[blank_index] = "________"
#             question = " ".join(words)
#             quiz_questions.append({"question": question, "answer": answer})
#
#     return quiz_questions