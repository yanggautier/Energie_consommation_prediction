from __future__ import annotations
import __main__
import bentoml
import pandas as pd
import cloudpickle
from pydantic_core import PydanticCustomError

from pydantic import BaseModel, ValidationError, field_validator

from typing import Optional, List
from bentoml.io import JSON
from transformers import (
    UsageCountTransformer,
    DistrictDensityTransformer,
    FeatureEngineeringTransformer,
    TypeConversionTransformer,
    PropertyTypeTransformer,
    CategoricalEncoderTransformer,
    NumericStandardScalerTransformer)

__main__.UsageCountTransformer = UsageCountTransformer
__main__.DistrictDensityTransformer = DistrictDensityTransformer
__main__.FeatureEngineeringTransformer = FeatureEngineeringTransformer
__main__.TypeConversionTransformer = TypeConversionTransformer
__main__.PropertyTypeTransformer = PropertyTypeTransformer
__main__.CategoricalEncoderTransformer = CategoricalEncoderTransformer
__main__.NumericStandardScalerTransformer = NumericStandardScalerTransformer


def load_pipeline(filename):
    with open(filename, 'rb') as f:
        return cloudpickle.load(f)


class Building(BaseModel):
    DataYear: int = 2016
    BuildingType: str
    PrimaryPropertyType: str
    SecondLargestPropertyUseType: Optional[str]
    ThirdLargestPropertyUseType: Optional[str]
    ZipCode: str
    CouncilDistrictCode: int
    Neighborhood: str
    YearBuilt: int
    NumberofBuildings: int
    NumberofFloors: int
    PropertyGFATotal: float
    PropertyGFAParking: float
    ListOfAllPropertyUseTypes: str
    LargestPropertyUseType: str

    @field_validator('CouncilDistrictCode')
    @classmethod
    def validate_x(cls, CouncilDistrictCode: int) -> int:
        if CouncilDistrictCode < 0 or CouncilDistrictCode > 0:
            raise PydanticCustomError(
                'the_answer_error',
                '{number} doit être plus grand que 0 et plus petit que 8!',
                {'number': CouncilDistrictCode},
            )
        return CouncilDistrictCode
    
    @field_validator('PropertyGFAParking')
    @classmethod
    def validate_x(cls, PropertyGFAParking: float) -> float:
        if PropertyGFAParking < 0:
            raise PydanticCustomError(
                'the_answer_error',
                '{number} doit être plus grand que 0!',
                {'number': PropertyGFAParking},
            )
        return PropertyGFAParking
    
    @field_validator('PropertyGFATotal')
    @classmethod
    def validate_x(cls, PropertyGFATotal: float) -> float:
        if PropertyGFATotal < 0:
            raise PydanticCustomError(
                'the_answer_error',
                '{number} doit être plus grand que 0!',
                {'number': PropertyGFATotal},
            )
        return PropertyGFATotal
    

    @field_validator('YearBuilt')
    @classmethod
    def validate_x(cls, YearBuilt: int) -> int:
        if YearBuilt < 0:
            raise PydanticCustomError(
                'the_answer_error',
                '{number} doit être plus grand que 0!',
                {'number': YearBuilt},
            )
        return YearBuilt
    

    class Config:
        arbitrary_types_allowed = True


class BuildingList(BaseModel):
    buildings: List[Building]

    class Config:
        arbitrary_types_allowed = True


@bentoml.service(name="energy_consumation_predictor")
class BuildingPredictorService:
    def __init__(self):
        # Lire les fichiers de modèle, pipeline, et standarscaler dans les attributs
        self.pipeline = load_pipeline("preprocessing.pipeline")

        self.model_energy = pd.read_pickle("model/lasso_energy.model")
        self.model_ges = pd.read_pickle("model/lasso_ges.model")
        
        self.y_scaler_energy = pd.read_pickle("scaler/y_energy.scaler")
        self.y_scaler_ges = pd.read_pickle("scaler/y_ges.scaler")

    @bentoml.api
    def predict_single(self, input_data: Building) -> dict:
        #try:
        # Transformer le données de requête en Pandas DataFrame

        building_dict = input_data.model_dump()
        building_df = pd.DataFrame([building_dict])
        
        transformed_data = self.pipeline.transform(building_df)

        # Prédiction de consommation d'énergie 
        energy_use_scaled = self.model_energy.predict(transformed_data)
        energy_use_final = round(float(self.y_scaler_energy.inverse_transform(energy_use_scaled.reshape(-1, 1)).flatten()[0]), 2)

        # Prédiction d'émission d'effet de serre
        total_ges_scaled = self.model_ges.predict(transformed_data)
        total_ges_final = round(float(self.y_scaler_ges.inverse_transform(total_ges_scaled.reshape(-1, 1)).flatten()[0]), 2)


        return {"prediction": {
                    "consommation d'énergie (kBtu)": energy_use_final,
                    "émission d'effet de serre": total_ges_final
                    }, 
                "status_code": 200}
        #except Exception as e:
        #    return {"error": str(e), "status_code": 500}

    @bentoml.api
    def predict_list(self, input_data: BuildingList) -> dict:
        try:
            # Transformer la liste de bâtiments en DataFrame
            buildings_dict = [building.model_dump() for building in input_data.buildings]
            buildings_df = pd.DataFrame(buildings_dict)

            # Appliquer les transformations
            transformed_data = self.pipeline.transform(buildings_df)

            # Prédiction de consommation d'énergie 
            energy_use_scaled = self.model_energy.predict(transformed_data)
            energy_use_final = self.y_scaler_energy.inverse_transform(energy_use_scaled.reshape(-1, 1)).flatten()

            # Prédiction d'émission d'effet de serre
            total_ges_scaled = self.model_ges.predict(transformed_data)
            total_ges_final = self.y_scaler_ges.inverse_transform(total_ges_scaled.reshape(-1, 1)).flatten()


            # Créer un dictionnaire de résultats avec l'index du bâtiment comme clé
            predictions = {
                f"building_{i}": {
                    "consommation d'énergie (kBtu)": round(float(energy_use_final[i]), 2),
                    "émission d'effet de serre": round(float(total_ges_final[i]),2)}
                for i in range(len(buildings_df))
            }

            return {
                "predictions": predictions,
                "status_code": 200,
                "number_of_predictions": len(predictions)
            }

        except Exception as e:
            return {"error": str(e), "status_code": 500}
