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
    # ListOfAllPropertyUseTypes: str
    LargestPropertyUseType: str
    SteamUsekBtu: float
    NaturalGasTherms: float
    DefaultData: bool
    ComplianceStatus: str

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
    
    @field_validator('SteamUsekBtu')
    @classmethod
    def validate_x(cls, SteamUsekBtu: float) -> float:
        if SteamUsekBtu < 0:
            raise PydanticCustomError(
                'the_answer_error',
                '{number} doit être plus grand que 0!',
                {'number': SteamUsekBtu},
            )
        return SteamUsekBtu
    
    @field_validator('NaturalGasTherms')
    @classmethod
    def validate_x(cls, NaturalGasTherms: float) -> float:
        if NaturalGasTherms < 0:
            raise PydanticCustomError(
                'the_answer_error',
                '{number} doit être plus grand que 0!',
                {'number': NaturalGasTherms},
            )
        return NaturalGasTherms

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
        self.model = pd.read_pickle("model/xgb.model")
        self.pipeline = load_pipeline("preprocessing.pipeline")
        self.scaler_y = pd.read_pickle("y.scaler")

    @bentoml.api
    def predict_single(self, input_data: Building) -> dict:
        try:
            # Transformer le données de requête en Pandas DataFrame
            building_dict = input_data.model_dump()
            building_df = pd.DataFrame([building_dict])
            transformed_data = self.pipeline.transform(building_df)

            # Prédiction de données
            result = self.model.predict(transformed_data)
            final_result = float(self.scaler_y.inverse_transform(result.reshape(-1, 1)).flatten()[0])

            return {"prediction": final_result, "status_code": 200}
        except Exception as e:
            return {"error": str(e), "status_code": 500}

    @bentoml.api
    def predict_list(self, input_data: BuildingList) -> dict:
        try:
            # Transformer la liste de bâtiments en DataFrame
            buildings_dict = [building.model_dump() for building in input_data.buildings]
            buildings_df = pd.DataFrame(buildings_dict)

            # Appliquer les transformations
            transformed_data = self.pipeline.transform(buildings_df)

            # Prédiction pour tous les bâtiments
            results = self.model.predict(transformed_data)
            final_results = self.scaler_y.inverse_transform(results.reshape(-1, 1)).flatten()

            # Créer un dictionnaire de résultats avec l'index du bâtiment comme clé
            predictions = {
                f"building_{i}": float(pred)
                for i, pred in enumerate(final_results)
            }

            return {
                "predictions": predictions,
                "status_code": 200,
                "number_of_predictions": len(predictions)
            }

        except Exception as e:
            return {"error": str(e), "status_code": 500}
