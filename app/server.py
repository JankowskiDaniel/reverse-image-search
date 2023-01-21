import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from models import Status

app = FastAPI()


@app.get("/", response_model = Status)
def check_status() -> JSONResponse:
    return JSONResponse({"connection": True})

@app.get("/text/match/image")
def text_match_image(query: str) -> JSONResponse:
    pass

@app.get("/image/match/image")
def image_match_image(image):
    pass

@app.get("image/match/text")
def describe_image(image):
    pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3020, reload=True)
