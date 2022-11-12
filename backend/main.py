import uvicorn
from fastapi import FastAPI, File, UploadFile
from functions import read_imagefile, classifier, parser, vqa

app = FastAPI()


@app.post("/classifier")
async def classifier_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    return classifier(image)


@app.post("/parser")
async def parser_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    return parser(image)


@app.post("/vqa")
async def vqa_api(file: UploadFile = File(...), question: str = None):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    return vqa(image, question)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=6969, debug=True)