from typing import List, Tuple
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse

from models import Status
from search_engine import SearchEngine
import numpy as np
from pydantic import BaseModel
search_engine = SearchEngine()
import cv2

app = FastAPI()

class Shape(BaseModel):
    shape: Tuple[int, int, int]



@app.get("/", response_model = Status)
def check_status() -> JSONResponse:
    return JSONResponse({"connection": True})

@app.get("/query_to_image/")
def text_match_image(query: str) -> JSONResponse:
    scores, matches = search_engine.match_query_to_image(query)
    return JSONResponse({"matches": matches,
                         "scores": scores})

@app.post("/similar_images/")
async def image_match_image(file: UploadFile) -> JSONResponse:
    data = await file.read()
    image = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    scores, similar_images = search_engine.find_similar_images(image)
    return JSONResponse({"matches": similar_images,
                         "scores": scores})

@app.post("/describe_image/")
async def describe_image(file: UploadFile) -> JSONResponse:
    data = await file.read()
    image = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    scores, descriptions = search_engine.match_image_to_texts(image)
    return JSONResponse({"matches": descriptions,
                         "scores": scores})

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=3020, reload=True)
