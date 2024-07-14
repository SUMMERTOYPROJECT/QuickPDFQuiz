
from src.app.service.pdf_service import PDFService
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from src.app.config.dependencies import get_pdf_service
import logging
router = APIRouter()

logger = logging.getLogger(__name__)

#-- 서버 상태 체크
@router.get("/check")
async def health_check():
    return {"message": "It's Working On NickController Service!"}

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), pdf_service: PDFService = Depends(get_pdf_service)):
    try:
        result = await pdf_service.process_pdf(file)
        return {"message": "PDF processed successfully", "result": result}
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))