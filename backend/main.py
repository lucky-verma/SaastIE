import uvicorn
from fastapi import FastAPI, File, UploadFile
from functions import read_imagefile, classifier, parser, vqa

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the SaasTie API!"}

@app.post("/classifier")
async def classifier_api(file: bytes = File()):
    if not file:
        return "Image must be jpg or png format!"
    image = read_imagefile(file)
    return classifier(image)


@app.post("/parser")
async def parser_api(file: bytes = File()):
    if not file:
        return "Image must be jpg or png format!"
    image = read_imagefile(file)
    return parser(image)


@app.post("/vqa")
async def vqa_api(question: str = "What's the total amount?", file: bytes = File()):
    print(question)
    if not file:
        return "Image must be jpg or png format!"
    if not question:
        return "Question must be provided!"
        
    image = read_imagefile(file)
    return vqa(image, question)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=6969, debug=True)