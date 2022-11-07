import uvicorn
from fastapi import FastAPI, File, UploadFile
from functions import read_imagefile, classifier

app = FastAPI()

@app.post("/classifier")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    return classifier(image)

if __name__ == "__main__":
    uvicorn.run(app, debug=True)