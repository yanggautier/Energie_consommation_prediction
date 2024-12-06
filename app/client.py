import bentoml

EXAMPLE_INPUT = {
    "DataYear": 2016, 
    "BuildingType": "NonResidential",
    "PrimaryPropertyType": "Other",
    "SecondLargestPropertyUseType": None,
    "ThirdLargestPropertyUseType": None,
    "ZipCode": "98101",
    "CouncilDistrictCode": 7,
    "Neighborhood": "DOWNTOWN",
    "YearBuilt": 2004,
    "NumberofBuildings": 1,
    "NumberofFloors": 11,
    "PropertyGFATotal": 299070.0,
    "PropertyGFAParking": 68432.0,
    # "ListOfAllPropertyUseTypes": "Other",
    "LargestPropertyUseType": "Other",
    "SteamUsekBtu": 0.0,
    "NaturalGasTherms": 346853.3125,
    "DefaultData": False,
    "ComplianceStatus": "Compliant"
}

    
def prediction_emission(building, url):
    """
        Test de précition sur la consommation d'un seul batîment
    """
    try:
        with bentoml.SyncHTTPClient(url) as client:
            response = client.predict_single(building)        
 

        if response["status_code"] == 200 and "prediction" in response:
            return response
        else:
            raise Exception("Erreur de connection au niveau de serveur !")

    except Exception as e:
        raise Exception("Erreur de connection")

if __name__ == '__main__':
    response = prediction_emission(EXAMPLE_INPUT, "http://localhost:8080")
    print(response)