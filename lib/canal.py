import json
import time
import matplotlib.pyplot as plt
import shutil
from loguru import logger
import numpy as np
from progress.bar import Bar
import scienceplots

plt.style.use("science")
# ignore where warnings
# np.seterr(divide='ignore', invalid='ignore')

TOL_WET_CELLS = 1e-12

TOL9 = 1e-9

DEBUG = False

P0 = 0.0  # Reference pressure in Pa in the boundary conditions


class Canal:
    """Class to simulate the evolution of a canal with a trapezoidal cross section
    imputs from a json file, the canal can be initialized with a dam break or a soliton
    online plotting is available, the canal can be simulated with a flux or wave scheme
    """

    def __init__(self, test=False, canal=None, case=None):

        self.gravity = 9.81  # Gravity acceleration in m/s^2
        self.rho = 1  # Water density in kg/m^3

        self.real_time = 0.0  # Real time of the simulation
        self.t_int = 0  # Time step counter

        if test:
            config = canal
        else:
            path_config = "config/canal.json"
            with open(path_config, "r") as f:
                config = json.load(f)

        length = config["length"]  # Length of the canal in m
        self.length = length

        delta_x = config["delta x"]
        self.dx = delta_x

        self.n = int(length / delta_x)

        width = config[
            "base width"
        ]  # Width of the canal in m, can be "file" for a file configuration
        self.width = width

        z_bed = config["z bed"]  # Bed of the canal in m
        self.z_mode = z_bed
        self.z = z_bed

        angle = config["angle"]  # Slope of the trapezoidal canal
        self.angle = angle * np.pi / 180  # Convert to radians

        manning = config["manning"]  # Manning coefficient
        self.manning = manning

        mode = config["initial mode"]  # Can be 'DAMBREAK' or 'SOLITON'
        self.mode = mode
        self.scheme = "flux"

        entropy_fix = config["entropy fix"]
        self.activate_entropy_fix = True if entropy_fix == "True" else False
        self.check_fix = False  # default value for no entropy fix

        cfl = config["CFL"]
        self.cfl = cfl

        end_time = config["end time"]
        self.end_time = end_time

        out_freq = config["output freq"]
        self.out_freq = out_freq
        self.out_counter = self.out_freq

        self.dt_list = []  # List of time steps

        self.x = np.linspace(0, length, self.n)

        if not test:
            self.prog_bar = Bar("Processing", max=self.end_time / self.out_freq)

        # set the width of the canal
        if width == "file":
            with open("config/width.txt", "r") as f:
                aux = np.array([float(x) for x in f.read().split()])
                # Check if the width file has the same length as the canal if not resample
                if len(aux) != self.n:
                    aux = np.interp(
                        np.linspace(0, self.length, self.n),
                        np.linspace(0, self.length, len(aux)),
                        aux,
                    )
                self.width = aux
        else:
            self.width = np.ones(self.n) * width

        # set the bed of the canal
        if z_bed == "file":
            with open("config/z_bed.txt", "r") as f:
                aux = np.array([float(x) for x in f.read().split()])
                # Check if the bed file has the same length as the canal if not resample
                if len(aux) != self.n:
                    aux = np.interp(
                        np.linspace(0, self.length, self.n),
                        np.linspace(0, self.length, len(aux)),
                        aux,
                    )
                self.z = aux
        else:
            self.z = np.ones(self.n) * z_bed

        logger.info(f"Canal created with {self.n} cells")
        logger.info(f"Initial mode set to {mode}")
        match mode:
            case "DAMBREAK":
                if test:
                    dam_break = case
                else:
                    with open("config/dambreak.json", "r") as f:
                        dam_break = json.load(f)

                position = dam_break["position"]
                left_height = dam_break["left height"]
                right_height = dam_break["right height"]
                right_u = dam_break["right u"]
                left_u = dam_break["left u"]
                right_w = dam_break["right w"]
                left_w = dam_break["left w"]
                exact = dam_break["exact"]

                self.position = position
                self.left_height = left_height
                self.right_height = right_height
                self.right_u = right_u
                self.left_u = left_u
                self.right_w = right_w
                self.left_w = left_w
                self.exact = True if exact == "True" else False

                # Initialize the canal with the dam break configuration
                self.h = np.zeros(self.n)
                self.hu = np.zeros(self.n)
                self.u = np.zeros(self.n)
                self.w = np.zeros(self.n)

                self.h[: int(position / self.dx)] = left_height
                self.h[int(position / self.dx) :] = right_height
                self.u[: int(position / self.dx)] = left_u
                self.u[int(position / self.dx) :] = right_u
                self.w[: int(position / self.dx)] = left_w
                self.w[int(position / self.dx) :] = right_w
                self.hu = self.h * self.u
                self.hw = self.h * self.w
                self.p = (
                    self.rho * self.gravity * self.h
                )  # initial pressure set to hydrostatic

                if exact:
                    with open("config/exact_dmb.txt", "r") as f:
                        self.exact_x, self.exact_h, _, self.exact_u = np.loadtxt(
                            f, unpack=True
                        )

            case "SOLITON":
                if test:
                    soliton = case
                else:
                    with open("config/soliton.json", "r") as f:
                        soliton = json.load(f)

                Height0 = soliton["H0"]
                Amplitude = soliton["A"]
                Xi = soliton["Xi"]
                Position = soliton["position"]
                self.wave_position = Position
                self.Height0 = Height0
                self.Amplitude = Amplitude
                self.Xi = Xi
                self.wave_celerity = np.sqrt(self.gravity * (Height0 + Amplitude))

                # Vector of the initial conditions
                self.h = self.h_sw_function(self.x, 0)
                self.u = self.u_sw_function()
                self.w = self.w_sw_function(self.x, 0)
                self.pnh_sw_function(self.x, 0)

            case "MCDONALD":
                # TO BE IMPLEMENTED
                pass

        ## initialize plots
        if not test:
            plt.ion()
            self.fig, self.ax = plt.subplots(2, 2, figsize=(10, 8))
            self.plot_results()  # plot initial conditions

    # Calule te hydraulic radius of the trapezoidal canal in terms of B and angle
    def _calc_hidraulic_radius(self):
        """Calculate the hydraulic radius of the canal in terms of the depth and the width of the canal, rectangular channel is considered"""
        # self.R = self.h*(self.width +self.h/np.tan(self.angle))/(self.width + 2*self.h/np.sin(self.angle))
        # R = np.where(self.h>TOL_wet_cells,(2*self.h+self.width)/(self.h*self.width),0)
        self.R = np.divide(
            2 * self.h + self.width,
            self.h * self.width,
            where=self.h > TOL_WET_CELLS,
            out=np.zeros_like(self.h),
        )

    # Calculate the friction slope of the canal
    def _calc_S_manning(self):
        """Calculate the friction slope of the canal in terms of the manning coefficient and the hydraulic radius of the canal"""
        # S_f = np.where(self.h>TOL_wet_cells,(self.u*self.manning)**2/(self.R**(4/3)),0)
        self.S_f = np.divide(
            (self.u * self.manning) ** 2,
            self.R ** (4 / 3),
            where=self.h > TOL_WET_CELLS,
            out=np.zeros_like(self.h),
        )

    def _calc_S0(self, A=None):
        """'Calculate the source term of the canal, energetic aproach is used, can be set to the source term model with the A argument"""
        dz = np.roll(self.z, -1) - self.z
        d = self.h + self.z
        dz_var = np.ones(self.n) * np.nan
        dz_var = np.where((dz > 0) & (d < np.roll(self.z, -1)), self.h, dz_var)
        dz_var = np.where(
            (dz < 0) & (np.roll(d, -1) < self.z), np.roll(self.h, -1), dz_var
        )
        dz_var = np.where(np.isnan(dz_var), dz, dz_var)
        h_var = np.where(dz > 0, self.h, np.roll(self.h, -1))
        Tb1 = -self.gravity * (h_var - np.abs(dz_var) / 2) * dz_var
        Tb2 = -self.gravity * (self.h + np.roll(self.h, -1)) * dz / 2

        if A is None:
            hu2 = self.hu * self.u
            u2 = self.u**2
            # A = ((np.roll(hu2,-1)-hu2) - self.h * (np.roll(u2,-1)-u2)/2)/(Tb1+Tb2)
            A = np.divide(
                (np.roll(hu2, -1) - hu2) - self.h * (np.roll(u2, -1) - u2) / 2,
                Tb1 + Tb2,
                where=(Tb1 + Tb2) > TOL_WET_CELLS,
                out=np.zeros_like(Tb1),
            )

        Tb3 = (1 - A) * Tb2 + A * Tb1
        self.S0 = Tb3

    ############################ VARIABLES CALCULATION FOR ROE SCHEME ############################
    def _calc_h_wall(self):
        self.h_wall = np.zeros(self.n)
        for i in range(self.n - 1):
            self.h_wall[i] = (self.h[i] + self.h[i + 1]) / 2
        self.h_wall[-1] = self.h[-1]  # Last value is the same as the previous one

    # c average velocity of the canal
    def _calc_c_wall(self):
        self.c_wall = np.zeros(self.n)
        for i in range(self.n - 1):
            self.c_wall[i] = np.sqrt(
                self.gravity * self.h_wall[i]
            )  # This formulation could need adjustments for width
        self.c_wall[-1] = self.c[-1]  # Last value is the same as the previous one

    def _calc_u_wall(self):
        self.u_wall = np.zeros(self.n)
        for i in range(self.n - 1):
            if self.wet_dry[i] == 1:
                self.u_wall[i] = (
                    self.u[i] * np.sqrt(self.h[i])
                    + self.u[i + 1] * np.sqrt(self.h[i + 1])
                ) / (np.sqrt(self.h[i]) + np.sqrt(self.h[i + 1]))
            else:
                self.u_wall[i] = 0
        self.u_wall[-1] = self.u[-1]  # Last value is the same as the previous one

    def _calc_lambdas(self):
        self.lambda1 = self.u_wall - self.c_wall
        self.lambda2 = self.u_wall + self.c_wall

    def _calc_cell_lambda(self):
        self.l1 = np.zeros(self.n)
        self.l2 = np.zeros(self.n)
        self.c = np.sqrt(self.gravity * self.h)
        for i in range(self.n):
            self.l1[i] = self.u[i] - self.c[i]
            self.l2[i] = self.u[i] + self.c[i]

    def _clac_alphas(self):
        self.alpha1 = np.zeros(self.n)
        self.alpha2 = np.zeros(self.n)
        for i in range(self.n - 1):
            if self.wet_dry[i] == 1:
                self.alpha1[i] = (
                    self.lambda2[i] * (self.h[i + 1] - self.h[i])
                    - self.u[i + 1] * self.h[i + 1]
                    + self.u[i] * self.h[i]
                ) / (self.width[i] * 2 * self.c_wall[i])
                self.alpha2[i] = (
                    -self.lambda1[i] * (self.h[i + 1] - self.h[i])
                    + self.u[i + 1] * self.h[i + 1]
                    - self.u[i] * self.h[i]
                ) / (self.width[i] * 2 * self.c_wall[i])
        self.alpha1[-1] = self.alpha1[-2]
        self.alpha2[-1] = self.alpha2[-2]

    def _calc_betas(self):
        self.beta1 = np.zeros(self.n)
        self.beta2 = np.zeros(self.n)
        for i in range(self.n - 1):
            if self.wet_dry[i] == 1:
                Sf_wall = (self.S_f[i] + self.S_f[i + 1]) / 2
                S0_wall = self.S0[i]
                self.beta1[i] = (
                    self.gravity * self.h_wall[i] * (Sf_wall) * self.dx - S0_wall
                ) / (
                    2 * self.c_wall[i]
                )  # will fail with non constant width
                self.beta2[i] = -self.beta1[i]
        self.beta1[-1] = self.beta1[-2]
        self.beta2[-1] = self.beta2[-2]

    def _calc_gammas(self):
        self.gamma1 = np.zeros(self.n)
        self.gamma2 = np.zeros(self.n)
        for i in range(self.n - 1):
            if self.wet_dry[i] == 1:
                self.gamma1[i] = self.alpha1[i] - self.beta1[i] / self.lambda1[i]
                self.gamma2[i] = self.alpha2[i] - self.beta2[i] / self.lambda2[i]
            else:
                self.gamma1[i] = 0
                self.gamma2[i] = 0

        self.gamma1[-1] = self.gamma1[-2]
        self.gamma2[-1] = self.gamma2[-2]

    # Check if the entropy fix is needed, only decompose the flow if the eigenvalues have different signs
    def check_entropy(self):
        self.check_fix = False
        for i in range(len(self.lambda1) - 1):
            if self.lambda1[i] * self.lambda1[i + 1] < 0:
                self.check_fix = True
                break
            if self.lambda2[i] * self.lambda2[i + 1] < 0:
                self.check_fix = True
                break
        # print log
        if DEBUG:
            logger.warning(
                f"Entropy fix activated at time {self.real_time} in cell {i}"
            )
        return self.check_fix

    # Calculate all vectors if there is a change in the eigenvalues on the current time step
    def entropy_fix(self):
        # Entropy fix following Morales-Hernandez, 2014
        self.lambda1_e = np.zeros(self.n)
        self.lambda2_e = np.zeros(self.n)

        for i in range(self.n - 1):
            if self.h[i] < TOL_WET_CELLS or self.h[i + 1] < TOL_WET_CELLS:
                pass
            else:
                if self.l1[i] * self.l1[i + 1] > 0:
                    # default values go to bar variables, 0 otherwise
                    self.lambda1[i] = self.lambda1[i]

                else:  # Entropy fix will define all pair of values
                    COC1 = (self.l1[i + 1] - self.lambda1[i]) / (
                        self.l1[i + 1] - self.l1[i]
                    )
                    self.lambda1_e[i] = self.lambda1[i] - self.l1[i] * COC1
                    self.lambda1[i] = self.l1[i] * COC1

                if self.l2[i] * self.l2[i + 1] > 0:
                    self.lambda2[i] = self.lambda2[i]
                else:
                    COC2 = (self.lambda2[i] - self.l2[i]) / (
                        self.l2[i + 1] - self.l2[i]
                    )
                    self.lambda2_e[i] = self.lambda2[i] - self.l2[i] * COC2
                    self.lambda2[i] = self.l2[i] * COC2

        # Last value is the same as the previous one
        self.lambda1[-1] = self.lambda1[-2]
        self.lambda2[-1] = self.lambda2[-2]
        self.lambda1_e[-1] = self.lambda1_e[-2]
        self.lambda2_e[-1] = self.lambda2_e[-2]

    # Friction fix
    def friction_fix(self):
        beta_z = -0.5 * self.gravity * self.h_wall * (np.roll(self.z, -1) - self.z)

    def calc_dt(self):
        aux = np.concatenate((self.lambda1, self.lambda2))
        self.dt = (
            self.cfl * self.dx / np.max(np.abs(aux))
        )  # Courant-Friedrichs-Lewy condition, limit the time step to the CFL condition
        # output and end limits
        if self.real_time + self.dt > self.out_counter:
            self.dt = self.out_counter - self.real_time
            self.out_counter += self.out_freq
            try:
                self.prog_bar.next()
            except:
                pass
        if self.real_time + self.dt > self.end_time:
            self.dt = self.end_time - self.real_time
        self.dt_list.append(self.dt)

    def update_hydro_waves(self):

        hls = (
            self.h
            + self.alpha1
            - np.divide(
                self.beta1,
                self.lambda1,
                where=self.wet_dry > TOL_WET_CELLS,
                out=np.zeros_like(self.lambda1),
            )
        )
        hrs = (
            np.roll(self.h, -1)
            - self.alpha2
            + np.divide(
                self.beta2,
                self.lambda2,
                where=self.wet_dry > TOL_WET_CELLS,
                out=np.zeros_like(self.lambda2),
            )
        )

        hls = np.where(np.abs(hls) > TOL_WET_CELLS, hls, 0)
        hrs = np.where(np.abs(hrs) > TOL_WET_CELLS, hrs, 0)

        hu_wave_minus = np.zeros(self.n)
        h_wave_minus = np.zeros(self.n)
        hu_wave_plus = np.zeros(self.n)
        h_wave_plus = np.zeros(self.n)

        # # # vector to store the fluxes

        for i in range(self.n - 1):
            if self.h[i + 1] < TOL_WET_CELLS and hrs[i] < TOL_WET_CELLS:
                h_wave_minus[i] = (
                    self.lambda1[i] * self.gamma1[i] + self.lambda2[i] * self.gamma2[i]
                )
            elif self.h[i] < TOL_WET_CELLS and hls[i] < TOL_WET_CELLS:
                h_wave_plus[i] = (
                    self.lambda1[i] * self.gamma1[i] + self.lambda2[i] * self.gamma2[i]
                )
            else:
                hu_wave_minus[i] = (
                    self.lambda1_minus[i] * self.gamma1[i] * self.lambda1[i]
                    + self.lambda2_minus[i] * self.gamma2[i] * self.lambda2[i]
                )
                h_wave_minus[i] = (
                    self.lambda1_minus[i] * self.gamma1[i]
                    + self.lambda2_minus[i] * self.gamma2[i]
                )
                hu_wave_plus[i] = (
                    self.lambda1_plus[i] * self.gamma1[i] * self.lambda1[i]
                    + self.lambda2_plus[i] * self.gamma2[i] * self.lambda2[i]
                )
                h_wave_plus[i] = (
                    self.lambda1_plus[i] * self.gamma1[i]
                    + self.lambda2_plus[i] * self.gamma2[i]
                )

        if self.check_fix:
            self.lambda1_e_minus = np.where(self.lambda1_e < 0, self.lambda1_e, 0)
            self.lambda2_e_minus = np.where(self.lambda2_e < 0, self.lambda2_e, 0)
            self.lambda1_e_plus = np.where(self.lambda1_e > 0, self.lambda1_e, 0)
            self.lambda2_e_plus = np.where(self.lambda2_e > 0, self.lambda2_e, 0)
            hu_wave_minus += (
                self.lambda1_e_minus * self.alpha1 * self.lambda1
                + self.lambda2_e_minus * self.alpha2 * self.lambda2
            )
            h_wave_minus += (
                self.lambda1_e_minus * self.alpha1 + self.lambda2_e_minus * self.alpha2
            )
            hu_wave_plus += (
                self.lambda1_e_plus * self.alpha1 * self.lambda1
                + self.lambda2_e_plus * self.alpha2 * self.lambda2
            )
            h_wave_plus += (
                self.lambda1_e_plus * self.alpha1 + self.lambda2_e_plus * self.alpha2
            )

        self.h -= self.dt / self.dx * (h_wave_minus + np.roll(h_wave_plus, 1))
        self.hu -= self.dt / self.dx * (hu_wave_minus + np.roll(hu_wave_plus, 1))

        # contorn conditions
        self.h[0] = self.left_height
        self.h[-1] = self.right_height
        self.hu[0] = self.left_u * self.left_height
        self.hu[-1] = self.right_u * self.right_height

        # update w
        u = np.divide(
            self.hu, self.h, where=self.h > TOL_WET_CELLS, out=np.zeros_like(self.h)
        )
        u_plus = 0.5 * (u + np.abs(u))
        u_minus = 0.5 * (np.roll(u, -1) - np.abs(np.roll(u, -1)))
        new_w = self.w - self.dt / self.dx * (
            u_plus * (self.w - np.roll(self.w, 1))
            + u_minus * (np.roll(self.w, -1) - self.w)
        )
        new_w[0] = new_w[1]
        new_w[-1] = new_w[-2]
        self.w = new_w

    def update_hydro_flux(self):

        right_front = np.where(
            (self.z != np.roll(self.z, -1)) & (np.roll(self.z, -1) > self.h + self.z),
            0,
            1,
        )
        left_front = np.where(
            (self.z != np.roll(self.z, -1))
            & (np.roll(self.z, -1) + np.roll(self.h, -1) < self.z),
            0,
            1,
        )

        # # # vector to store the fluxes
        hu_flux_minus = (
            self.lambda1_minus * self.gamma1 * self.lambda1
            + self.lambda2_minus * self.gamma2 * self.lambda2
        )
        h_flux_minus = (
            self.lambda1_minus * self.gamma1 + self.lambda2_minus * self.gamma2
        )
        hu_flux_plus = (
            -self.lambda1_plus * self.gamma1 * self.lambda1
            - self.lambda2_plus * self.gamma2 * self.lambda2
        )
        h_flux_plus = -self.lambda1_plus * self.gamma1 - self.lambda2_plus * self.gamma2

        if self.check_fix:
            self.lambda1_e_minus = np.where(self.lambda1_e < 0, self.lambda1_e, 0)
            self.lambda2_e_minus = np.where(self.lambda2_e < 0, self.lambda2_e, 0)
            self.lambda1_e_plus = np.where(self.lambda1_e > 0, self.lambda1_e, 0)
            self.lambda2_e_plus = np.where(self.lambda2_e > 0, self.lambda2_e, 0)
            hu_flux_minus += (
                self.lambda1_e_minus * self.alpha1 * self.lambda1
                + self.lambda2_e_minus * self.alpha2 * self.lambda2
            )
            h_flux_minus += (
                self.lambda1_e_minus * self.alpha1 + self.lambda2_e_minus * self.alpha2
            )
            hu_flux_plus -= (
                self.lambda1_e_plus * self.alpha1 * self.lambda1
                + self.lambda2_e_plus * self.alpha2 * self.lambda2
            )
            h_flux_plus -= (
                self.lambda1_e_plus * self.alpha1 + self.lambda2_e_plus * self.alpha2
            )

        for f in [hu_flux_minus, h_flux_minus, hu_flux_plus, h_flux_plus]:
            f *= right_front * left_front

        # update fluxes with physical fluxes
        hu_flux_minus += self.hu * self.h + 0.5 * self.gravity * self.h**2
        h_flux_minus += self.hu
        hu_flux_plus += (
            np.roll(self.hu, -1) * np.roll(self.h, -1)
            + 0.5 * self.gravity * np.roll(self.h, -1) ** 2
        )
        h_flux_plus += np.roll(self.hu, -1)

        self.h -= self.dt / self.dx * (h_flux_minus - np.roll(h_flux_plus, 1))
        self.hu -= self.dt / self.dx * (hu_flux_minus - np.roll(hu_flux_plus, 1))

        # contorn conditions
        self.h[0] = self.left_height
        self.h[-1] = self.right_height
        self.hu[0] = self.left_u * self.left_height
        self.hu[-1] = self.right_u * self.right_height

        # update w
        hu_dw = (
            self.hu
            + self.lambda1_minus * self.gamma1
            + self.lambda2_minus * self.gamma2
        )  # h flux minus
        hw_flux_minus = np.where(hu_dw > 0, hu_dw * self.w, hu_dw * np.roll(self.w, -1))  # w transport
        self.hw = self.hw - self.dt / self.dx * (
            hw_flux_minus - np.roll(hw_flux_minus, 1) 
        )  # update hw with (huw)⁻_{i+1/2} - (huw)⁻_{i-1/2}
        self.hw[0] = self.hw[1]
        self.hw[-1] = self.hw[-2]

        self.w = np.divide(
            self.hw, self.h, where=self.h > TOL_WET_CELLS, out=np.zeros_like(self.h)
        )

    # Calculate variable vectors
    def calc_vectors(self):

        # Calculate cell variables
        self._calc_hidraulic_radius()
        self._calc_S_manning()
        self._calc_S0()  # Can set A as an argument to switch between source terms model
        self._calc_cell_lambda()

        # Calculate wall variables
        self._calc_h_wall()
        self._calc_c_wall()
        self._calc_u_wall()
        self._calc_lambdas()
        self._clac_alphas()
        self._calc_betas()
        self._calc_gammas()

        if self.activate_entropy_fix:
            a = self.check_entropy()
            if a:
                self.entropy_fix()

        self.lambda1_minus = np.where(self.lambda1 < 0, self.lambda1, 0)
        self.lambda2_minus = np.where(self.lambda2 < 0, self.lambda2, 0)
        self.lambda1_plus = np.where(self.lambda1 > 0, self.lambda1, 0)
        self.lambda2_plus = np.where(self.lambda2 > 0, self.lambda2, 0)

    def check_wet_dry(self):
        # 1 if wet, 0 if dry
        self.wet_dry = np.zeros(self.n)
        for i in range(self.n - 1):
            self.wet_dry[i] = (
                1 if (self.h[i] > TOL_WET_CELLS or self.h[i + 1] > TOL_WET_CELLS) else 0
            )
            if self.wet_dry[i] == 0:
                self.hu[i] = 0
                self.hu[i + 1] = 0

    def update_cell_values(self):
        self.u = np.divide(
            self.hu, self.h, where=self.h > TOL_WET_CELLS, out=np.zeros_like(self.h)
        )
        if self.scheme == "flux":
            self.w = np.divide(
                self.hw, self.h, where=self.h > TOL_WET_CELLS, out=np.zeros_like(self.h)
            )
        elif self.scheme == "wave":
            self.hw = self.h * self.w

    ##### Temporal loop #####
    def temporal_loop(self, mode="flux"):
        self.scheme = mode
        self.t_int = 1
        self.check_wet_dry()
        while self.real_time < self.end_time:
            self.calc_vectors()
            self.calc_dt()
            match mode:
                case "flux":
                    self.update_hydro_flux()
                case "wave":
                    self.update_hydro_waves()

            self.check_wet_dry()
            self.update_cell_values()

            # update non-hydrostatic variables
            self.non_hydrostatic_correction()

            self.real_time += self.dt
            if self.out_freq == "no output":
                continue
            else:
                if abs(self.real_time - self.out_counter + self.out_freq) < 1e-6:
                    # self.plot_TDMA()
                    self.plot_results()
                    # time.sleep(0.01)
                self.t_int += 1
                if self.t_int % 100 == 0:
                    if self.check_fix:
                        print("\n")
                        logger.warning(f"Entropy fix at Time: {self.real_time:.2f} s")

    # Plot the results
    def plot_results(self):
        for ax_row in self.ax:
            for ax in ax_row:
                ax.clear()
                ax.grid()
        x = self.x
        self.ax[0, 0].plot(x, self.h + self.z, label="h")
        self.ax[0, 0].set_title("h")
        # self.ax[0].set_ylim(min(self.left_height,self.right_height)-0.1,max(self.left_height,self.right_height)+0.1)
        self.ax[1, 0].plot(x, self.u, label="u")
        self.ax[1, 0].set_title("u")
        # Fr =np.where(self.h>TOL_wet_cells,self.u/np.sqrt(self.gravity*self.h),0)
        Fr = np.divide(
            self.u,
            np.sqrt(self.gravity * self.h),
            where=self.h > TOL_WET_CELLS,
            out=np.zeros_like(self.h),
        )

        self.ax[0, 1].plot(x, self.w, label="Fr")

        self.ax[0, 1].set_title("W")
        self.ax[1, 1].plot(x, self.p, label="P")
        self.ax[1, 1].set_title("Pressure")
        # plot z bed solid
        if self.z_mode == "file":
            self.ax[0, 0].fill_between(x, 0, self.z, color="black", alpha=0.2)

        # Exact solution if available
        if self.exact:
            self.ax[0].plot(
                self.exact_x, self.exact_h, ".", ms=1, color="black", label="h exact"
            )
            self.ax[1].plot(
                self.exact_x, self.exact_u, ".", ms=1, color="black", label="u exact"
            )

        self.fig.suptitle(f"Time: {self.real_time:.2f} s")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.savefig(f"img/state_{self.t_int}_{self.scheme}.png")

    def plot_TDMA(self):
        for ax_row in self.ax:
            for ax in ax_row:
                ax.clear()
                ax.grid()
        x = self.x
        self.ax[0, 0].plot(x, self.A_sub_diag, label="h")
        self.ax[0, 0].set_title("A")

        self.ax[1, 0].plot(x, self.B_diag, label="u")
        self.ax[1, 0].set_title("B")

        self.ax[0, 1].plot(x, self.C_sup_diag, label="Fr")

        self.ax[0, 1].set_title("C")
        self.ax[1, 1].plot(x, self.D, label="P")
        self.ax[1, 1].set_title("D")
        # plot z bed solid
        if self.z_mode == "file":
            self.ax[0, 0].fill_between(x, 0, self.z, color="black", alpha=0.2)

        # Exact solution if available
        if self.exact:
            self.ax[0].plot(
                self.exact_x, self.exact_h, ".", ms=1, color="black", label="h exact"
            )
            self.ax[1].plot(
                self.exact_x, self.exact_u, ".", ms=1, color="black", label="u exact"
            )

        self.fig.suptitle(f"Time: {self.real_time:.2f} s")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.savefig(f"img/state_{self.t_int}_{self.scheme}.png")

    ### elliptic non-hydrostatic model ###
    def Tridiagonal(self):
        """
        Calculate the coefficients of the tridiagonal matrix for the elliptic non-hydrostatic model.

        Returns:
            None
        """
        self.A_sub_diag = np.zeros(self.n)
        self.B_diag = np.zeros(self.n)
        self.C_sup_diag = np.zeros(self.n)
        self.D = np.zeros(self.n)  # termino independiente

        # Auxiliar variables
        phi = 0.5 * (
            np.roll(self.h, -1)
            + 2 * np.roll(self.z, -1)
            - np.roll(self.h, 1)
            - 2 * np.roll(self.z, 1)
        )  # mean phi of two walls [i+1]-[i-1]
        phi_edge = (
            self.h + 2 * self.z - np.roll(self.h, 1) - 2 * np.roll(self.z, 1)
        )  # mean phi of two cells [i]-[i-1]
        h_edge = 0.5 * (self.h + np.roll(self.h, 1))  # mean h of two cells [i]-[i-1]
        q_edge = 0.5 * (self.hu + np.roll(self.hu, 1))  # mean q of two cells [i]-[i-1]

        # Calculate the coefficients of the tridiagonal matrix
        self.A_sub_diag = (np.roll(phi, 1) - 2 * np.roll(self.h, 1)) * (
            phi_edge + 2 * h_edge
        )
        self.B_diag = (
            16 * self.dx**2
            + phi_edge * (np.roll(phi, 1) + phi + 2 * np.roll(self.h, 1) - 2 * self.h)
            + 2 * h_edge * (np.roll(phi, 1) - phi + 4 * h_edge)
        )
        self.C_sup_diag = (phi + 2 * self.h) * (phi_edge - 2 * h_edge)
        self.D = -(
            4
            * self.dx
            / self.dt
            * (
                self.h * (self.hu - np.roll(self.hu, 1))
                - q_edge * phi_edge
                + 2 * self.dx * h_edge * self.w
            )
        )

        self.A_sub_diag[0] = self.A_sub_diag[1]  # CONTOUR
        self.B_diag[0] = self.B_diag[1]
        self.C_sup_diag[0] = self.C_sup_diag[1]
        self.D[0] = self.D[1]

        self.A_sub_diag[-1] = self.A_sub_diag[-2]  # CONTOUR
        self.B_diag[-1] = self.B_diag[-2]
        self.C_sup_diag[-1] = self.C_sup_diag[-2]
        self.D[-1] = self.D[-2]

    def TDMA(self):
        """
        Solves a tridiagonal linear system using the Thomas algorithm (also known as the TDMA algorithm).

        This method solves a tridiagonal linear system of equations of the form Ax = d, where A is a tridiagonal matrix,
        x is the unknown vector, and d is the right-hand side vector.

        Returns:
            None

        Raises:
            None
        """

        a = self.A_sub_diag
        b = self.B_diag
        c = self.C_sup_diag
        d = self.D

        r = np.zeros(self.n)
        rp = np.zeros(self.n)
        bp = np.zeros(self.n)

        for i in range(2, self.n - 1):
            r[i] = d[i]

        r[1] = d[1] - a[1] * P0
        rp[1] = r[1]
        bp[1] = b[1]

        for i in range(2, self.n):
            if abs(bp[i - 1]) > TOL9:
                bp[i] = b[i] - a[i] * c[i - 1] / bp[i - 1]
                rp[i] = r[i] - rp[i - 1] * a[i] / bp[i - 1]
            else:
                bp[i] = 0
                rp[i] = 0

        if abs(bp[-1]) > TOL9:
            self.p[-1] = rp[-1] / bp[-1]
        else:
            self.p[-1] = 0

        self.p[0] = P0

        for i in range(self.n - 2, 0, -1):
            if abs(bp[i]) > TOL9:
                self.p[i] = (rp[i] - c[i] * self.p[i + 1]) / bp[i]
            else:
                self.p[i] = 0
        # print(self.p[497:504])
        # input()

    def non_hydrostatic_correction(self):
        self.Tridiagonal()
        self.TDMA()
        # self.p[0] = self.p[1]  # self.rho*self.gravity*self.h[0]
        # self.p[-1] = self.p[-2]  # self.rho*self.gravity*self.h[-1]
        self.p = 0.7 * self.p
        # print(f'\n dt:{self.real_time:.3}, pmax:{max(self.p)}')

        # # # update hu
        # self.hu -= (
        #     self.dt
        #     / self.dx
        #     * (
        #         self.h * (np.roll(self.p, 1) - self.p)
        #         + (np.roll(self.p, 1) + self.p)
        #         * (np.roll(self.h, -1) - np.roll(self.h, 1)
        #         + 2 * (np.roll(self.z, -1) - np.roll(self.z, 1)))
        #         / 4
        #     )
        # )

        # # # update w
        # h_edge = 0.5 * (self.h + np.roll(self.h, 1))
        # self.w += self.dt * 2 * self.p / h_edge

    # SOLITON
    def h_sw_function(self, x, t):
        return (
            self.Height0
            + self.Amplitude
            / np.cosh(
                (x - self.wave_celerity * t)
                / self.Height0
                * np.sqrt(self.Xi * self.Amplitude / (self.Height0 + self.Amplitude))
            )
            ** 2
        )

    def u_sw_function(self):
        return self.wave_celerity * (1 - self.Height0 / self.h)

    def w_sw_function(self, x, t):
        return (
            -self.wave_celerity
            * self.Amplitude
            / self.h
            * np.sqrt(self.Xi * self.Amplitude / (self.Height0 + self.Amplitude))
            * np.tanh(
                (x - self.wave_celerity * t)
                / self.Height0
                * np.sqrt(self.Xi * self.Amplitude / (self.Height0 + self.Amplitude))
            )
            / np.cosh(
                (x - self.wave_celerity * t)
                / self.Height0
                * np.sqrt(self.Xi * self.Amplitude / (self.Height0 + self.Amplitude))
            )
            ** 2
        )

    def pnh_sw_function(self, x, t):
        return (
            (self.wave_celerity * self.Amplitude) ** 2
            / (2 * (self.Amplitude + self.Height0) * self.h)
            * (
                2
                * self.Amplitude
                / self.h
                / np.cosh(
                    (x - self.wave_celerity * t)
                    / self.Height0
                    * np.sqrt(
                        self.Xi * self.Amplitude / (self.Height0 + self.Amplitude)
                    )
                )
                ** 6
                - 3
                / np.cosh(
                    (x - self.wave_celerity * t)
                    / self.Height0
                    * np.sqrt(
                        self.Xi * self.Amplitude / (self.Height0 + self.Amplitude)
                    )
                )
                * np.tanh(
                    (x - self.wave_celerity * t)
                    / self.Height0
                    * np.sqrt(
                        self.Xi * self.Amplitude / (self.Height0 + self.Amplitude)
                    )
                )
                ** 2
                + np.sinh(
                    (x - self.wave_celerity * t)
                    / self.Height0
                    * np.sqrt(
                        self.Xi * self.Amplitude / (self.Height0 + self.Amplitude)
                    )
                )
                ** 2
                + 1
                / np.cosh(
                    (x - self.wave_celerity * t)
                    / self.Height0
                    * np.sqrt(
                        self.Xi * self.Amplitude / (self.Height0 + self.Amplitude)
                    )
                )
                ** 2
            )
        )

    def save_config(self, name=None):
        # zip the config folder into test_cases
        if name is None:
            name = f"test_case_{self.mode}_{self.scheme}"
        shutil.make_archive(f"cases/{name}", "zip", "config")
