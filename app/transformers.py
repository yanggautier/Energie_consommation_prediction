import cloudpickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

class UsageCountTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['CountUsage'] = X_copy.apply(
            lambda row: sum(1 for col in ['PrimaryPropertyType', 'SecondLargestPropertyUseType', 'ThirdLargestPropertyUseType']
                          if row[col]), axis=1
        )
        return X_copy.drop(columns=['SecondLargestPropertyUseType', 'ThirdLargestPropertyUseType'])

class DistrictDensityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.district_density = None
    
    def fit(self, X, y=None):
        self.district_density = X.groupby('CouncilDistrictCode')['PropertyGFATotal'].mean()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['DistrictDensity'] = X_copy['CouncilDistrictCode'].map(self.district_density)
        return X_copy
    

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy["BuildingAge"] = X_copy["DataYear"] - X_copy["YearBuilt"]
        X_copy["GFAPerFloor"] = (X_copy["PropertyGFATotal"] / X_copy["NumberofFloors"]).round(2)
        X_copy["GFAPerBuilding"] = (X_copy["PropertyGFATotal"] / X_copy["NumberofBuildings"]).round(2)
        X_copy["ParkingGFARate"] = (X_copy["PropertyGFAParking"] / X_copy["PropertyGFATotal"]).round(2)
        return X_copy

class TypeConversionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy["ZipCode"] = X_copy["ZipCode"].astype(int).astype(str)
        X_copy["CouncilDistrictCode"] = X_copy["CouncilDistrictCode"].astype(str)
        X_copy["YearBuilt"] = X_copy["YearBuilt"].astype(str)
        return X_copy.drop(columns=["DataYear"])

class PropertyTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        
    def fit(self, X, y=None):
        property_types = X["ListOfAllPropertyUseTypes"].str.split(',').apply(lambda x: [tag.strip() for tag in x])
        self.mlb.fit(property_types)
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        property_types = X_copy["ListOfAllPropertyUseTypes"].str.split(',').apply(lambda x: [tag.strip() for tag in x])
        tags_encoded = pd.DataFrame(
            self.mlb.transform(property_types),
            columns=self.mlb.classes_,
            index=X_copy.index
        )
        X_copy["TotalUseType"] = len(property_types)
        X_copy = X_copy.drop(["ListOfAllPropertyUseTypes"], axis=1)
        return pd.concat([X_copy, tags_encoded], axis=1)

class CategoricalEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}
        self.all_feature_names = {}
        
    def fit(self, X, y=None):
        for column in X.select_dtypes(["object"]).columns:
            encoder = OneHotEncoder(handle_unknown='ignore')  # Ajout de handle_unknown='ignore'
            self.encoders[column] = encoder
            self.encoders[column].fit(X[column].values.reshape(-1, 1))
            # Stocker les noms des features pour chaque colonne
            self.all_feature_names[column] = self.encoders[column].get_feature_names_out([column])
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for column in self.encoders.keys():
            if column in X_copy.columns:
                # Transformation pour les colonnes présentes
                encoded = self.encoders[column].transform(X_copy[column].values.reshape(-1, 1)).toarray()
                feature_names = self.all_feature_names[column]
            else:
                # Création d'une matrice de zéros pour les colonnes manquantes
                encoded = np.zeros((len(X_copy), len(self.all_feature_names[column])))
                feature_names = self.all_feature_names[column]
            
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_copy.index)
            X_copy = pd.concat([X_copy, encoded_df], axis=1)
            
            # Ne supprimer la colonne que si elle existe
            if column in X_copy.columns:
                X_copy = X_copy.drop(column, axis=1)
        
        return X_copy.fillna(0)
    
class NumericStandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        # Sélectionner uniquement les colonnes numériques
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        self.numeric_columns = numeric_columns
        self.scaler.fit(X[numeric_columns])
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        # Standardiser uniquement les colonnes numériques
        if len(self.numeric_columns) > 0:
            X_copy[self.numeric_columns] = self.scaler.transform(X_copy[self.numeric_columns])
        return X_copy