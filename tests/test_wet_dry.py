import sys
import os

lib_path = os.path.abspath("lib")
sys.path.append(lib_path)
# print(sys.path)
from canal import *
import numpy as np

canal = {
    "length": 20,
    "delta x": 0.04,
    "base width": 1.0,
    "z bed": 0.0,
    "angle": 90,
    "manning": 0.00,
    "initial mode": "DAMBREAK",
    "entropy fix": "True",
    "CFL": 0.9,
    "end time": 0.8,
    "output freq": 1,
    "Non Hydrostatic": "False"
}

dambreak = {
    "position": 10,
    "left height": 0.4,
    "right height": 0.0,
    "left u": 0,
    "right u": 0,
    "left w": 0.0,
    "right w": 0.0,
    "exact": "False",
}

n = int(canal["length"] / canal["delta x"])
h_results = [
    (
        dambreak["left height"]
        if i < n * dambreak["position"] / canal["length"]
        else dambreak["right height"]
    )
    for i in range(n)
]

z_bed = [
    (
        0.0
        if i < n * dambreak["position"] / canal["length"]
        else 0.5
    )
    for i in range(n)
]

h_results = np.array(h_results)
z_bed = np.array(z_bed)

def mean_error(h_results, h_exact):
    return np.sum(np.sqrt((h_results - h_exact) ** 2)) / len(h_results)


def test_dam_break_waves():
    river = Canal(test=True, canal=canal, case=dambreak)
    river.z = z_bed
    print(len(river.h),len(river.z))
    river.temporal_loop(mode="wave")
    error = mean_error(h_results, river.h)
    print("Error: ", error)
    # plt.plot(river.h)
    # plt.plot(h_results)
    # plt.show()

    assert error < 1e-3


def test_dam_break_fluxes():
    river = Canal(test=True, canal=canal, case=dambreak)
    river.z = z_bed
    river.temporal_loop(mode="flux")
    error = mean_error(h_results, river.h)
    assert error < 1e-3


if __name__ == "__main__":
    test_dam_break_waves()
    test_dam_break_fluxes()
    print("All tests passed!")
