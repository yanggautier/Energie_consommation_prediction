#!/bin/bash
bentoml serve service:BuildingPredictorService &
sleep 10
python -m pytest test.py