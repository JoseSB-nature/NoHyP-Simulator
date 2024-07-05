from lib.canal import *

canal = {
    "length": 20,
    "delta x": 0.05,
    "base width": 1.0,
    "z bed": 0.0,
    "angle": 90, 
    "manning": 0.00,
    "initial mode": "DAMBREAK",
    "entropy fix": "True",
    "CFL": 0.9,
    "end time": 0.8,
    "output freq": 1
}

dambreak = {
    "position": 10,
    "left height": 1.2,
    "right height": 1,
    "left u": 0,
    "right u": 0,
    "left w": 0.0,
    "right w": 0.0,
    "exact": "False"
}

def test_dam_break():