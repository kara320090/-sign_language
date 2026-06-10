from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router as predict_router

app = FastAPI(
    title="Sign Language Backend",
    description="수어 인식, 표현 강도 분석, LLM 후처리 통합 백엔드",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "backend server is running"
    }