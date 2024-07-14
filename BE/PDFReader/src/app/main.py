import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from py_eureka_client import eureka_client
from src.app.controller import pdf_controller
import uvicorn
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app_: FastAPI):
    # Startup
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    host_ip = "localhost"
    logger.info(f"Initializing Eureka client with host: {host_ip}, port: 8000")
    await eureka_client.init_async(
        eureka_server="http://localhost:8761/eureka",
        app_name="fastapi-pdf-service",
        instance_port=8000,
        instance_host=host_ip,
        instance_ip=host_ip
    )
    logger.info("Eureka client initialized")
    yield
    # Shutdown
    logger.info("Stopping Eureka client")
    await eureka_client.stop_async()

app = FastAPI(lifespan=lifespan)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 구체적인 오리진을 지정하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Received request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Returned response: Status {response.status_code}")
    return response

app.include_router(pdf_controller.router)
if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)