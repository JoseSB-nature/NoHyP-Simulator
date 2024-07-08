from lib.canal import *
import numpy as np

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

h_results = [1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.200000, 1.158540, 1.101944, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.097682, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]

def mean_error(h_results, h_exact):
    return np.sum(np.sqrt((h_results - h_exact)**2))/len(h_results)


def test_dam_break_waves():
    river = Canal(canal, dambreak)
    river.temporal_loop(mode='wave')
    mean_error(h_results, river.h)

    assert mean_error(h_results, river.h) < 1e-3

def test_dam_break_fluxes():
    river = Canal(canal, dambreak)
    river.temporal_loop(mode='flux')
    mean_error(h_results, river.h)

    assert mean_error(h_results, river.h) < 1e-3
