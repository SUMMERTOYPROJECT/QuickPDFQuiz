# src/app/controller/pdf_controller.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from src.app.service.pdf_service import PDFService
from src.app.config.dependencies import get_pdf_service
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/check")
async def health_check():
    return {"message": "It's Working On PDF Controller Service!"}


@router.post("/upload-pdf")
async def upload_pdf(
        file: UploadFile = File(...),
        pdf_service: PDFService = Depends(get_pdf_service)
):
    try:
        result = await pdf_service.save_pdf(file)
        return JSONResponse(content={
            "message": "PDF upload successful. Processing started in background.",
            "file_id": result["file_id"]
        })
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during PDF upload")


@router.get("/status/{file_id}")
async def get_processing_status(
        file_id: str,
        pdf_service: PDFService = Depends(get_pdf_service)
):
    status_info = await pdf_service.get_processing_status(file_id)

    if status_info["status"] == "Not Found":
        raise HTTPException(status_code=404, detail=status_info["message"])
    elif status_info["status"] == "Expired":
        raise HTTPException(status_code=410, detail=status_info["message"])

    return JSONResponse(content=status_info)


@router.post("/search")
async def search_in_pdfs(
        query: str,
        file_id: str = None,
        pdf_service: PDFService = Depends(get_pdf_service)
):
    try:
        results = await pdf_service.search_vectors(query, file_id)
        return JSONResponse(content={"results": results})
    except Exception as e:
        logger.error(f"Error searching PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error occurred during PDF search")

@router.get("/generate-quiz/{file_id}")
async def generate_quiz(
    file_id: str,
    num_questions: int = Query(default=5, ge=1, le=10),
    pdf_service: PDFService = Depends(get_pdf_service)
):
    try:
        quiz = await pdf_service.generate_quiz(file_id, num_questions)
        return JSONResponse(content=quiz)
    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        raise HTTPException(status_code=500, detail="Error occurred during quiz generation")
