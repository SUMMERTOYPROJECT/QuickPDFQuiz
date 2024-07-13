import logging
from fastapi import FastAPI
from py_eureka_client import eureka_client
import uvicorn
import asyncio
import socket

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    await eureka_client.init_async(
        eureka_server="http://host.docker.internal:8761/eureka",
        app_name="fastapi-pdf-service",
        instance_port=8000,
        instance_host=host_ip,
        instance_ip=host_ip
    )

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/health")
async def health_check():
    return {"status": "UP"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)