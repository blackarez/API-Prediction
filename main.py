from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Configuración CORS para permitir solicitudes desde tu aplicación web (ajusta según sea necesario)
origins = [
    "http://localhost",  # Reemplaza tu_puerto_de_desarrollo con el puerto real
    "https://analisis-datos-b9f5c.web.app",  # Agrega aquí el dominio de tu aplicación en producción
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    ano: int
    year: int
    

@app.get("/")
async def root():
    model = load('model_decisionTreeRegressor.joblib')
    model_response = model.predict([[4988.705582, 985.881631]])
    return {"message": model_response.tolist()}

@app.post("/predict")
def predict(item: Item):
    model = load('model_decisionTreeRegressor.joblib')
    data = [
        item.ano,
        item.year
    ]

    model_response = model.predict([data])
    return {"message": model_response.tolist()}

@app.get("/test")
async def root():
    model = load('model_LinearRegression.joblib')
    model_response = model.predict([[4988.705582, 985.881631]])
    return {"message": model_response.tolist()}

@app.post("/predictB")
def predict(item: Item):
    model = load('model_LinearRegression.joblib')
    data = [
        item.ano,
        item.year
    ]

    model_response = model.predict([data])
    return {"message": model_response.tolist()}
