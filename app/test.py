import os
import bentoml
import unittest
from service import BuildingList


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


class TestConnection(unittest.TestCase):
    def setUp(self):
        """Configuration initiale avant chaque test"""
        self.uri = os.getenv('http://localhost:3000')

    
    def test_prediction_single(self):
        """
            Test de précition sur la consommation d'un seul batîment
        """
        try:
            with bentoml.SyncHTTPClient(self.uri, timeout=60) as client:
                result = client.predict_single(EXAMPLE_INPUT)        
        except Exception as e:
            self.fail(e)

        self.assertTrue(result["status_code"] == 200)
        self.assertTrue("prediction" in result)
        self.assertTrue(type(result["prediction"]) == float)
        self.assertTrue(result["prediction"] > 0)

    def test_prediction_list(self):
        """
            Test de précition sur la consommation d'une liste de batîment
        """
        try:
            with bentoml.SyncHTTPClient(self.uri, timeout=60) as client:
                building_list = BuildingList(buildings=[EXAMPLE_INPUT])
                result = client.predict_list(building_list)  
        except Exception as e:
            self.fail(e)
        
        self.assertTrue(result["status_code"] == 200)
        self.assertTrue("predictions" in result)
        self.assertTrue("building_0" in result["predictions"])
        self.assertTrue(type(result["predictions"]["building_0"]) == float)
        self.assertTrue(result["predictions"]["building_0"] > 0)
        self.assertTrue("number_of_predictions" in result)
        self.assertEqual(result["number_of_predictions"], 1)

if __name__ == '__main__':
    unittest.main()