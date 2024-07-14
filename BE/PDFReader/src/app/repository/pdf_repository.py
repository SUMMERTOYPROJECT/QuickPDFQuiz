from sqlalchemy.orm import Session
from src.app.models.pdf import PDF
from src.app.dto.pdf_schema import PDFCreate

class PDFRepository:
    def create_pdf(self, db: Session, pdf: PDFCreate):
        db_pdf = PDF(filename=pdf.filename)
        db.add(db_pdf)
        db.commit()
        db.refresh(db_pdf)
        return db_pdf

    def get_pdf(self, db: Session, pdf_id: int):
        return db.query(PDF).filter(PDF.id == pdf_id).first()