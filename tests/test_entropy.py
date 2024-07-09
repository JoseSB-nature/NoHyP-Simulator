import sys
import os
lib_path = os.path.abspath('lib')
sys.path.append(lib_path)
# print(sys.path)
from canal import *
import numpy as np

canal = {
    "length": 20,
    "delta x": 0.02,
    "base width": 1.0,
    "z bed": 0.0,
    "angle": 90, 
    "manning": 0.00,
    "initial mode": "DAMBREAK",
    "entropy fix": "True",
    "CFL": 0.9,
    "end time": 0.4,
    "output freq": 1
}

dambreak = {
    "position": 10,
    "left height": 10,
    "right height": 1,
    "left u": 0,
    "right u": 0,
    "left w": 0.0,
    "right w": 0.0,
    "exact": "False"
}

h_results = [10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 10.00000, 9.933109, 9.866104, 9.799327, 9.732776, 9.666451, 9.600354, 9.534483, 9.468839, 9.403422, 9.338232, 9.273268, 9.208531, 9.144021, 9.079737, 9.015681, 8.951851, 8.888248, 8.824871, 8.761722, 8.698799, 8.636103, 8.573633, 8.511391, 8.449375, 8.387586, 8.326023, 8.264688, 8.203579, 8.142697, 8.082041, 8.021613, 7.961411, 7.901436, 7.841688, 7.782166, 7.722872, 7.663804, 7.604962, 7.546348, 7.487960, 7.429799, 7.371865, 7.314157, 7.256677, 7.199423, 7.142396, 7.085595, 7.029022, 6.972675, 6.916554, 6.860661, 6.804994, 6.749554, 6.694341, 6.639355, 6.584595, 6.530063, 6.475756, 6.421677, 6.367825, 6.314199, 6.260800, 6.207627, 6.154682, 6.101963, 6.049471, 5.997206, 5.945167, 5.893355, 5.841770, 5.790412, 5.739281, 5.688376, 5.637698, 5.587247, 5.537022, 5.487024, 5.437253, 5.387709, 5.338392, 5.289301, 5.240437, 5.191800, 5.143390, 5.095206, 5.047249, 4.999519, 4.952016, 4.904739, 4.857689, 4.810866, 4.764270, 4.717900, 4.671757, 4.625841, 4.580152, 4.534689, 4.489453, 4.444444, 4.399662, 4.355107, 4.310778, 4.266676, 4.222801, 4.179152, 4.135730, 4.092535, 4.049567, 4.006826, 3.964311, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 3.961748, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]

n = int(canal["length"]/canal["delta x"])
resampled_h_results = np.interp(np.linspace(0, canal["length"],n), np.linspace(0, canal["length"],len(h_results)), h_results)

h_results = resampled_h_results


# define index points

dambreak_index = int(dambreak["position"]/canal["delta x"])
delta_index = int(n/50)

# print(dambreak_index, delta_index)

def mean_error(h_results:np.ndarray, h_exact:np.ndarray)->float:
    # return np.sum(np.sqrt((h_results - h_exact)**2))/len(h_results)
    return np.sum(np.abs(h_results - h_exact))/len(h_results)


def test_dam_break_waves():
    river = Canal(test=True, canal=canal, case=dambreak)
    river.temporal_loop(mode='wave')
    a = h_results[dambreak_index - delta_index:dambreak_index + delta_index]/np.max(h_results)
    b = river.h[dambreak_index - delta_index:dambreak_index + delta_index]/np.max(h_results)

    error = mean_error(a, b)
    print("Error: ", error)
    # plt.plot(river.h)
    # plt.plot(h_results) 
    # plt.show()

    assert error < 3e-3

def test_dam_break_fluxes():
    river = Canal(test=True, canal=canal, case=dambreak)
    river.temporal_loop(mode='flux')
    a = h_results[dambreak_index - delta_index:dambreak_index + delta_index]/np.max(h_results)
    b = river.h[dambreak_index - delta_index:dambreak_index + delta_index]/np.max(h_results)
    error = mean_error(a, b)
    assert error < 3e-3


if __name__ == '__main__':
    test_dam_break_waves()
    test_dam_break_fluxes()
    print("All tests passed!")