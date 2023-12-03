from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

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
    country: str


class ItemRL(BaseModel):
    ano: float
    country: float

listaTipeCancerUnic = np.array([ 8,  9, 24, 26, 19, 20, 23, 15, 18, 16, 11,  6, 13, 14,  2,  0,  3,
        5, 10,  7, 12, 22, 21,  4,  1, 25, 17])

def dataPrediction(country, ano):
    listaCountry = np.array(['Year', 'Tipo de Cancer', 'Code_AFG', 'Code_AGO', 'Code_ALB', 'Code_AND', 'Code_ANZ', 'Code_AR ', 'Code_ARE', 'Code_ARG', 'Code_ARM', 'Code_ASM', 'Code_ATG', 'Code_AUS', 'Code_AUT', 'Code_AZE', 'Code_BDI', 'Code_BEL', 'Code_BEN', 'Code_BFA', 'Code_BGD', 'Code_BGR', 'Code_BHR', 'Code_BHS', 'Code_BIH', 'Code_BLR', 'Code_BLZ', 'Code_BMU', 'Code_BOL', 'Code_BRB', 'Code_BRN', 'Code_BTN', 'Code_BWA', 'Code_CAF', 'Code_CAN', 'Code_CAR ', 'Code_CHE', 'Code_CHL', 'Code_CIV', 'Code_CMR', 'Code_COD', 'Code_COG', 'Code_COL', 'Code_COM', 'Code_CPV', 'Code_CRI', 'Code_CSSA', 'Code_CUB', 'Code_CYP', 'Code_CZE', 'Code_DJI', 'Code_DMA', 'Code_DNK', 'Code_DOM', 'Code_DZA', 'Code_ECU', 'Code_EGY', 'Code_ENG', 'Code_ERI', 'Code_ESP', 'Code_EST', 'Code_ETH', 'Code_FIN', 'Code_FJI', 'Code_FRA', 'Code_FSM', 'Code_GAB', 'Code_GBR', 'Code_GEO', 'Code_GHA', 'Code_GIN', 'Code_GMB', 'Code_GNB', 'Code_GNQ', 'Code_GRC', 'Code_GRD', 'Code_GRL', 'Code_GTM', 'Code_GUM', 'Code_GUY', 'Code_HND', 'Code_HRV', 'Code_HTI', 'Code_HUN', 'Code_IRL', 'Code_IRN', 'Code_IRQ', 'Code_ISL', 'Code_ISR', 'Code_ITA', 'Code_JAM', 'Code_JOR', 'Code_JPN', 'Code_KAZ', 'Code_KEN', 'Code_KGZ', 'Code_KHM', 'Code_KIR', 'Code_KOR', 'Code_KWT', 'Code_LAO', 'Code_LBN', 'Code_LBR', 'Code_LBY', 'Code_LCA', 'Code_LKA', 'Code_LSO', 'Code_LTU', 'Code_LUX', 'Code_LVA', 'Code_MAR', 'Code_MDA', 'Code_MDG', 'Code_MDV', 'Code_MHL', 'Code_MKD', 'Code_MLI', 'Code_MLT', 'Code_MMR', 'Code_MNE', 'Code_MNG', 'Code_MNP', 'Code_MOZ', 'Code_MRT', 'Code_MUS', 'Code_MWI', 'Code_MYS', 'Code_NAM', 'Code_NER', 'Code_NGA', 'Code_NIC', 'Code_NIR', 'Code_NLD', 'Code_NOR', 'Code_NPL', 'Code_NZL', 'Code_OC', 'Code_OMN', 'Code_PAK', 'Code_PAN', 'Code_PER', 'Code_PHL', 'Code_PNG', 'Code_POL', 'Code_PRI', 'Code_PRK', 'Code_PRT', 'Code_PRY', 'Code_PSE', 'Code_QAT', 'Code_ROU', 'Code_RWA', 'Code_SAU', 'Code_SCT', 'Code_SDN', 'Code_SEN', 'Code_SGP', 'Code_SLB', 'Code_SLE', 'Code_SLV', 'Code_SOM', 'Code_SRB', 'Code_SSD', 'Code_SSSA', 'Code_STP', 'Code_SUR', 'Code_SVK', 'Code_SVN', 'Code_SWE', 'Code_SWZ', 'Code_SYC', 'Code_SYR', 'Code_TCD', 'Code_TGO', 'Code_THA', 'Code_TJK', 'Code_TKM', 'Code_TLS', 'Code_TON', 'Code_TTO', 'Code_TUN', 'Code_TWN', 'Code_TZA', 'Code_UGA', 'Code_UKR', 'Code_URY', 'Code_UZB', 'Code_VCT', 'Code_VEN', 'Code_VIR', 'Code_VNM', 'Code_VUT', 'Code_WA', 'Code_WSM', 'Code_YEM', 'Code_ZAF', 'Code_ZMB', 'Code_ZWE'])
    entradas_prediccion = pd.DataFrame({'Code': [country]*len(listaTipeCancerUnic), 'Year': [ano]*len(listaTipeCancerUnic), 'Tipo de Cancer': listaTipeCancerUnic})
    entradas_prediccion = pd.get_dummies(entradas_prediccion, columns=['Code'], drop_first=False)
    entradas_prediccion = entradas_prediccion.reindex(columns=listaCountry, fill_value=0)    
    return entradas_prediccion

@app.get("/")
async def root():
    model = load('model_decisionTreeRegressorV2.joblib')
    model_response = model.predict(dataPrediction('COL', 2050))
    resultados = pd.DataFrame({'Tipo de Cancer': listaTipeCancerUnic, 'Probabilidad de Mortalidad': model_response})
    return {"message": resultados.to_json()}

@app.post("/predict")
def predict(item: Item):
    model = load('model_decisionTreeRegressorV2.joblib')
    model_response = model.predict(dataPrediction(item.country, item.ano))
    resultados = pd.DataFrame({'Tipo de Cancer': listaTipeCancerUnic, 'Probabilidad de Mortalidad': model_response})
    return {"message": resultados.to_json()}

@app.get("/test")
async def root():
    model = load('model_LinearRegression.joblib')
    model_response = model.predict([[4988.705582, 985.881631]])
    return {"message": model_response.tolist()}

@app.post("/predictB")
def predict(itemRL: ItemRL):
    model = load('model_LinearRegression.joblib')
    data = [
        itemRL.ano,
        itemRL.country
    ]

    model_response = model.predict([data])
    return {"message": model_response.tolist()}
