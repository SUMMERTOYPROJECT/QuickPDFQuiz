from pydantic import BaseModel
from datetime import datetime

class PDFBase(BaseModel):
    filename: str

class PDFCreate(PDFBase):
    pass

class PDFInDB(PDFBase):
    id: int
    content: str
    created_at: datetime

    class Config:
        orm_mode = True