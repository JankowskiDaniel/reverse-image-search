from typing import List
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from models import Status
from search_engine import SearchEngine
import numpy as np

search_engine = SearchEngine()

app = FastAPI()



@app.get("/", response_model = Status)
def check_status() -> JSONResponse:
    return JSONResponse({"connection": True})

@app.get("/query_to_image/")
def text_match_image(query: str) -> JSONResponse:
    matches = search_engine.match_query_to_image(query)
    return JSONResponse({"matches": matches})

@app.post("/similar_images")
def image_match_image(image: List[int]) -> JSONResponse:
    image = np.array(image)
    similar_images = search_engine.find_similar_images(image)
    return JSONResponse({"similar": similar_images})

@app.post("/describe_image")
def describe_image(image: List[int]) -> JSONResponse:
    image = np.array(image)
    descriptions = search_engine.match_image_to_texts(image)
    return JSONResponse({"descriptions": descriptions})

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=3020, reload=True)
