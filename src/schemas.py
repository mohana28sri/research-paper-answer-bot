from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question asked about research papers"
    )


class AnswerResponse(BaseModel):
    answer: str
    sources: list
    retrieved_chunks: int


class HealthResponse(BaseModel):
    status: str
    service: str
    llm_provider: str
    model: str
    database: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int