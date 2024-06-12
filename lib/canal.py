import numpy as np
from progress.bar import Bar
from loguru import logger
from numba import njit
import json
import time
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

DEBUG = False
class Canal:
    def __init__(self):
        
        self.gravity = 9.81 # Gravity acceleration in m/s^2
        
        self.real_time = 0.0 # Real time of the simulation
        self.t_int = 0 # Time step counter
        
        path_config = 'config/canal.json' 
        with open(path_config,'r') as f:
            config = json.load(f)
        
        length = config['length'] # Length of the canal in m
        self.length = length
        
        delta_x = config['delta x']
        self.dx = delta_x
        
        self.n = int(length/delta_x)
        
        width = config['base width'] # Width of the canal in m, can be "file" for a file configuration
        self.width = width
        
        z_bed = config['z bed'] # Bed of the canal in m
        self.z = z_bed
        
        angle = config['angle'] # Slope of the trapezoidal canal
        self.angle = angle*np.pi/180 # Convert to radians
        
        manning = config['manning'] # Manning coefficient
        self.manning = manning
        
        mode = config['initial mode'] # Can be 'DAMBREAK' or 'SOLITON'
        self.mode = mode
        
        entropy_fix = config['entropy fix']
        self.activate_entropy_fix = True if entropy_fix == 'True' else False
        self.check_fix = False # default value for no entropy fix
        
        CFL = config['CFL']
        self.CFL = CFL
        
        end_time = config['end time']
        self.end_time = end_time
        
        out_freq = config['output freq']
        self.out_freq = out_freq
        self.out_counter = self.out_freq
        
        self.dt_list = [] # List of time steps
        
        self.prog_bar = Bar('Processing', max=self.end_time/self.out_freq)
        
        # set the width of the canal
        if width == "file":
            with open('config/width.txt','r') as f:
                aux = np.array([float(x) for x in f.read().split()])
                # Check if the width file has the same length as the canal if not resample
                if len(aux) != self.n:
                    aux = np.interp(np.linspace(0,self.length,self.n),np.linspace(0,self.length,len(aux)),aux)
                self.width = aux
        else:
            self.width = np.ones(self.n)*width
            
        # set the bed of the canal
        if z_bed == "file":
            with open('config/z_bed.txt','r') as f:
                aux = np.array([float(x) for x in f.read().split()])
                # Check if the bed file has the same length as the canal if not resample
                if len(aux) != self.n:
                    aux = np.interp(np.linspace(0,self.length,self.n),np.linspace(0,self.length,len(aux)),aux)
                self.z = aux
        else:
            self.z = np.ones(self.n)*z_bed
        
        logger.info(f"Canal created with {self.n} cells")
        logger.info(f"Initial mode set to {mode}")
        match mode:
            case "DAMBREAK":
                with open('config/dambreak.json','r') as f:
                    dam_break = json.load(f)
                    
                position = dam_break['position']
                left_height = dam_break['left height']
                right_height = dam_break['right height']
                right_u = dam_break['right u']
                left_u = dam_break['left u']
                right_w = dam_break['right w']
                left_w = dam_break['left w']
                exact = dam_break['exact']
                
                self.position = position
                self.left_height = left_height
                self.right_height = right_height
                self.right_u = right_u
                self.left_u = left_u
                self.right_w = right_w
                self.left_w = left_w
                self.exact = True if exact == 'True' else False
                
                # Initialize the canal with the dam break configuration
                self.h = np.zeros(self.n)
                self.hu = np.zeros(self.n)
                self.u = np.zeros(self.n)
                self.w = np.zeros(self.n)
                
                self.h[:int(position/self.dx)] = left_height
                self.h[int(position/self.dx):] = right_height
                self.u[:int(position/self.dx)] = left_u
                self.u[int(position/self.dx):] = right_u
                self.w[:int(position/self.dx)] = left_w
                self.w[int(position/self.dx):] = right_w
                self.hu = self.h*self.u
                self.hw = self.h*self.w
                
                if exact:
                    with open('config/exact_dmb.txt','r') as f:
                        self.exact_x, self.exact_h, _, self.exact_u = np.loadtxt(f,unpack=True)
                        
                
            case "SOLITON":
                # TO BE IMPLEMENTED
                pass
            
            case "MCDONALD":
                # TO BE IMPLEMENTED
                pass
        
        ## initialize plots
        plt.ion()
        self.fig, self.ax = plt.subplots(3,1,figsize=(8,8))
        self.plot_results() # plot initial conditions
    
       
    # Calule te hydraulic radius of the trapezoidal canal in terms of B and angle
    def calc_hidraulic_radius(self):
        self.R = self.h*(self.width +self.h/np.tan(self.angle))/(self.width + 2*self.h/np.sin(self.angle))
        
    # Calculate the friction slope of the canal
    def calc_S_manning(self):
        self.S_f = (self.u*self.manning)**2/(self.R**(4/3))
      
############################ VARIABLES CALCULATION FOR ROE SCHEME ############################  
    def calc_h_wall(self):
        self.h_wall = np.zeros(self.n)
        for i in range(self.n-1):
            self.h_wall[i] = (self.h[i] + self.h[i+1])/2
        self.h_wall[-1] = self.h_wall[-2] # Last value is the same as the previous one    
        
    # c average velocity of the canal
    def calc_c_wall(self):
        self.c_wall = np.zeros(self.n)
        for i in range(self.n-1):
            self.c_wall[i] = np.sqrt(self.gravity*self.h_wall[i]) # This formulation could need adjustments for width
        self.c_wall[-1] = self.c_wall[-2] # Last value is the same as the previous one
            
    def calc_u_wall(self):
        self.u_wall = np.zeros(self.n)
        for i in range(self.n-1):
            self.u_wall[i] = (self.u[i]*np.sqrt(self.h[i]) + self.u[i+1]*np.sqrt(self.h[i+1]))/(np.sqrt(self.h[i]) + np.sqrt(self.h[i+1]))
        self.u_wall[-1] = self.u_wall[-2] # Last value is the same as the previous one
    
    def calc_lambdas(self):
        self.lambda1 = self.u_wall - self.c_wall
        self.lambda2 = self.u_wall + self.c_wall
        
    def calc_cell_lambda(self):
        self.l1 = np.zeros(self.n)
        self.l2 = np.zeros(self.n)
        for i in range(self.n):
            self.l1[i] = self.u[i] - np.sqrt(self.gravity*self.h[i])
            self.l2[i] = self.u[i] + np.sqrt(self.gravity*self.h[i])
            
            
    def clac_alphas(self):
        self.alpha1 = np.zeros(self.n)
        self.alpha2 = np.zeros(self.n)
        for i in range(self.n-1):
            self.alpha1[i] = (self.lambda2[i]*(self.h[i+1]-self.h[i]) - self.u[i+1]*self.h[i+1] + self.u[i]*self.h[i])/(self.width[i]*2*self.c_wall[i])
            self.alpha2[i] = (-self.lambda1[i]*(self.h[i+1]-self.h[i]) + self.u[i+1]*self.h[i+1] - self.u[i]*self.h[i])/(self.width[i]*2*self.c_wall[i])
        self.alpha1[-1] = self.alpha1[-2]
        self.alpha2[-1] = self.alpha2[-2]
        
    def calc_betas(self):
        self.beta1 = np.zeros(self.n)
        self.beta2 = np.zeros(self.n)
        for i in range(self.n-1):
            Sf_wall = (self.S_f[i] + self.S_f[i+1])/2
            S0_wall = (self.z[i] - self.z[i+1])/self.dx
            self.beta1[i] = -(self.gravity*self.h_wall[i]*(S0_wall-Sf_wall)*self.dx)/(2*self.c_wall[i]) # will fail with non constant width
            self.beta2[i] = -self.beta1[i]
        self.beta1[-1] = self.beta1[-2]
        self.beta2[-1] = self.beta2[-2]
        
    def calc_gammas(self):
        self.gamma1 = np.zeros(self.n)
        self.gamma2 = np.zeros(self.n)
        for i in range(self.n-1):
            self.gamma1[i] = self.alpha1[i]-self.beta1[i]/self.lambda1[i]
            self.gamma2[i] = self.alpha2[i]-self.beta2[i]/self.lambda2[i]

        self.gamma1[-1] = self.gamma1[-2]
        self.gamma2[-1] = self.gamma2[-2]

    # Check if the entropy fix is needed, only decompose the flow if the eigenvalues have different signs
    def check_entropy(self):
        self.check_fix = False
        for i in range(len(self.lambda1)-1):
            if self.lambda1[i]*self.lambda1[i+1] < 0:
                self.check_fix = True
                break
            if self.lambda2[i]*self.lambda2[i+1] < 0:
                self.check_fix = True
                break
        # print log
        if DEBUG: logger.warning(f"Entropy fix activated at time {self.real_time} in cell {i}")
        return self.check_fix
 
    # Calculate all vectors if there is a change in the eigenvalues on the current time step    
    def entropy_fix(self):
        # Entropy fix following Morales-Hernandez, 2014
        self.lambda1_hat = np.zeros(self.n)
        self.lambda2_hat = np.zeros(self.n)
        self.lambda1_bar = np.zeros(self.n)
        self.lambda2_bar = np.zeros(self.n)
        
        for i in range(self.n-1):
            if self.l1[i]*self.l1[i+1] > 0 and self.l2[i]*self.l2[i+1] > 0:
                # default values go to bar variables, 0 otherwise
                self.lambda1_bar[i] = self.lambda1[i]
                self.lambda2_bar[i] = self.lambda2[i]
                
            else: # Entropy fix will define all pair of values
                self.lambda1_bar[i] = self.l1[i]*(self.l1[i+1]-self.lambda1[i])/(self.l1[i+1]-self.l1[i])
                self.lambda2_bar[i] = self.l2[i]*(self.l2[i+1]-self.lambda2[i])/(self.l2[i+1]-self.l2[i])
                self.lambda1_hat[i] = self.l1[i+1]*(self.l1[i]-self.lambda1[i])/(self.l1[i]-self.l1[i+1])
                self.lambda2_hat[i] = self.l2[i+1]*(self.l2[i]-self.lambda2[i])/(self.l2[i]-self.l2[i+1])

        # Last value is the same as the previous one
        self.lambda1_bar[-1] = self.lambda1_bar[-2]
        self.lambda2_bar[-1] = self.lambda2_bar[-2]
        self.lambda1_hat[-1] = self.lambda1_hat[-2]
        self.lambda2_hat[-1] = self.lambda2_hat[-2]
    
    # Friction fix
    def friction_fix(self):
        beta_z = -0.5*self.gravity*self.h_wall*(np.roll(self.z,-1)-self.z)
 
    def calc_dt(self):
        aux = np.concatenate((self.lambda1,self.lambda2))
        self.dt = self.CFL*self.dx/np.max(np.abs(aux)) # Courant-Friedrichs-Lewy condition, limit the time step to the CFL condition
        # output and end limits
        if self.real_time + self.dt > self.out_counter:
            self.dt = self.out_counter - self.real_time
            self.out_counter += self.out_freq
            self.prog_bar.next()
        if self.real_time + self.dt > self.end_time:
            self.dt = self.end_time - self.real_time
        self.dt_list.append(self.dt)
        
    # update the hydrodynamic variables for the hydrostatic part    
    def update_hydro_contrib(self):
        
        hu_flux = self.hu + self.lambda1_minus*self.gamma1*self.lambda1 + self.lambda2_minus*self.gamma2*self.lambda2
        huw_flux = np.where(hu_flux>0, hu_flux*self.w, hu_flux*np.roll(self.w,-1))
        
        
        for i in range(1,self.n):
            h_contrib = 0
            hu_contrib = 0
            hw_contrib = 0
            if self.check_fix: # we use all the variables calculated in the entropy fix
                if self.lambda1_bar[i] > 0:
                    h_contrib += self.lambda1_bar[i-1]*self.gamma1[i-1]
                    hu_contrib += self.lambda1_bar[i-1]**2*self.gamma1[i-1]
                if self.lambda1_bar[i] < 0:
                    h_contrib += self.lambda1_bar[i]*self.gamma1[i]
                    hu_contrib += self.lambda1_bar[i]**2*self.gamma1[i]
                if self.lambda2_bar[i] > 0:
                    h_contrib += self.lambda2_bar[i-1]*self.gamma2[i-1]
                    hu_contrib += self.lambda2_bar[i-1]**2*self.gamma2[i-1]
                if self.lambda2_bar[i] < 0:
                    h_contrib += self.lambda2_bar[i]*self.gamma2[i]
                    hu_contrib += self.lambda2_bar[i]**2*self.gamma2[i]
                # if self.lambda1_hat[i] > 0:
                #     h_contrib += self.lambda1_hat[i-1]*self.alpha1[i-1] # betta hat are set to zero
                #     hu_contrib += self.lambda1_hat[i-1]**2*self.alpha1[i-1]
                # if self.lambda1_hat[i] < 0:
                #     h_contrib += self.lambda1_hat[i]*self.alpha1[i]
                #     hu_contrib += self.lambda1_hat[i]**2*self.alpha1[i]
                # if self.lambda2_hat[i] > 0:
                #     h_contrib += self.lambda2_hat[i-1]*self.alpha2[i-1]
                #     hu_contrib += self.lambda2_hat[i-1]**2*self.alpha2[i-1]
                # if self.lambda2_hat[i] < 0:
                #     h_contrib += self.lambda2_hat[i]*self.alpha2[i]
                #     hu_contrib += self.lambda2_hat[i]**2*self.alpha2[i]                    
            else:
                if self.lambda1[i-1] > 0:
                    h_contrib += self.lambda1[i-1]*self.gamma1[i-1] #update first component of the conservative vector
                    hu_contrib += self.lambda1[i-1]**2*self.gamma1[i-1] #update second component of the conservative vector
                if self.lambda1[i] < 0:
                    h_contrib += self.lambda1[i]*self.gamma1[i]
                    hu_contrib += self.lambda1[i]**2*self.gamma1[i]
                if self.lambda2[i-1] > 0:
                    h_contrib += self.lambda2[i-1]*self.gamma2[i-1]
                    hu_contrib += self.lambda2[i-1]**2*self.gamma2[i-1]
                if self.lambda2[i] < 0:
                    h_contrib += self.lambda2[i]*self.gamma2[i]
                    hu_contrib += self.lambda2[i]**2*self.gamma2[i]
                           
            self.h[i] = self.h[i] - self.dt/self.dx*(h_contrib)
            self.hu[i] = self.hu[i] - self.dt/self.dx*(hu_contrib)
        
        # update hw
        self.hw -= self.dt/self.dx*(np.roll(huw_flux,1) - huw_flux)
        
        # contorn conditions
        self.h[0] = self.left_height
        self.h[-1] = self.right_height
        self.hu[0] = self.left_u*self.left_height
        self.hu[-1] = self.right_u*self.right_height
               
    def update_hydro_flux(self):
        
        # # vector to store the fluxes
        # h_flux = self.h.copy()
        # hu_flux = self.hu.copy()
        
        # h_flux += self.lambda1_minus*self.gamma1*self.lambda1
        # h_flux += self.lambda2_minus*self.gamma2*self.lambda2
        
        # hu_flux += self.lambda1_minus*self.gamma1*self.lambda1**2
        # hu_flux += self.lambda2_minus*self.gamma2*self.lambda2**2
        
        h_flux_up = np.roll(self.hu,-1)
        h_flux_dw = self.hu
        hu_flux_up = np.roll(self.hu**2/self.h+0.5*self.gravity*self.h**2,-1)
        hu_flux_dw = self.hu**2/self.h+0.5*self.gravity*self.h**2
        
        for i in range(1,self.n):
            if self.lambda1[i] < 0:
                h_flux_dw[i] += self.lambda1[i]*self.gamma1[i]
                hu_flux_dw[i] += self.lambda1[i]**2*self.gamma1[i]
            else:
                h_flux_up[i] -= self.lambda1[i]*self.gamma1[i]
                hu_flux_up[i] -= self.lambda1[i]**2*self.gamma1[i]
            if self.lambda2[i] < 0:
                h_flux_dw[i] += self.lambda2[i]*self.gamma2[i]
                hu_flux_dw[i] += self.lambda2[i]**2*self.gamma2[i]
            else:
                h_flux_up[i] -= self.lambda2[i]*self.gamma2[i]
                hu_flux_up[i] -= self.lambda2[i]**2*self.gamma2[i]

        self.h -= self.dt/self.dx*(h_flux_dw-np.roll(h_flux_up,1))
        self.hu -= self.dt/self.dx*(hu_flux_dw-np.roll(hu_flux_up,1))                
        # self.h -= self.dt/self.dx*(np.roll(h_flux,1)-h_flux)
        # self.hu -= self.dt/self.dx*(np.roll(hu_flux,1)-hu_flux)
        # contorn conditions
        self.h[0] = self.left_height
        self.h[-1] = self.right_height
        self.hu[0] = self.left_u
        self.hu[-1] = self.right_u
               
    # Calculate variable vectors
    def debug_calc_vectors(self):
        self.u = self.hu/self.h
        t0 = time.time()
        self.calc_hidraulic_radius()
        t1 = time.time()
        
        self.calc_S_manning()
        t2 = time.time()
        
        self.calc_h_wall()
        self.calc_c_wall()
        self.calc_u_wall()
        t4 = time.time()
        
        self.calc_lambdas()
        self.calc_cell_lambda()
        self.clac_alphas()
        self.calc_betas()
        self.calc_gammas()
        t5 = time.time()
        
        self.calc_eigenvectors()
        t6 = time.time()
        
        if self.activate_entropy_fix:
            a = self.check_entropy()
            if a: 
                logger.info(f"Entropy fix activated")
                self.entropy_fix()
        t7 = time.time()
                # add time usage in % of total time
        total_time = t7-t0
        if self.check_fix: logger.debug(f"Total time: {total_time:.2f}\n\
            R:{(t1-t0)/total_time:.2f}; ({t1-t0})\n\
            S:{(t2-t1)/total_time:.2f}; ({t2-t1})\n\
            walls:{(t4-t2)/total_time:.2f}; ({t4-t2})\n\
            lambda:{(t5-t4)/total_time:.2f}; ({t5-t4})\n\
            eigenvectors:{(t6-t5)/total_time:.2f}; ({t6-t5})\n\
            entropy fix:{(t7-t6)/total_time:.2f}; ({t7-t6})\n") 

  # Calculate variable vectors
    def calc_vectors(self):
        
        self.u = self.hu/self.h
        self.w = self.hw/self.h
        self.calc_hidraulic_radius()
      
        self.calc_S_manning()
        
        self.calc_h_wall()
        self.calc_c_wall()
        self.calc_u_wall()
        
        self.calc_lambdas()
        self.calc_cell_lambda()
        self.clac_alphas()
        self.calc_betas()
        self.calc_gammas()
        
        
        self.lambda1_minus = np.where(self.lambda1 < 0, self.lambda1, 0)
        self.lambda2_minus = np.where(self.lambda2 < 0, self.lambda2, 0)
        
        if self.activate_entropy_fix:
            a = self.check_entropy()
            if a:
                self.entropy_fix()


    ##### Temporal loop #####
    def temporal_loop(self, mode = 'flux'):
        self.t_int = 1
        while self.real_time < self.end_time:
            self.calc_vectors()
            self.calc_dt()
            match mode:
                case 'flux':
                    self.update_hydro_flux()
                case 'cont':
                    self.update_hydro_contrib()
            self.real_time += self.dt
            if abs(self.real_time - self.out_counter + self.out_freq) < 1e-6:
                self.plot_results()
                # time.sleep(0.01)
            self.t_int += 1
            if self.t_int % 100 == 0:
                if self.check_fix: logger.info(f"Time: {self.real_time:.2f} s")


    # Plot the results
    def plot_results(self):
        for ax in self.ax:
            ax.clear()
            ax.grid()   
        x = np.linspace(0,self.length,self.n)
        self.ax[0].plot(x,self.h,label='h')
        self.ax[0].set_ylim(min(self.left_height,self.right_height)-0.1,max(self.left_height,self.right_height)+0.1)
        self.ax[1].plot(x,self.u,label='u')
        Fr = self.u/np.sqrt(self.gravity*self.h)
        self.ax[2].plot(x,self.w,label='w')
        
        # Exact solution if available
        if self.exact:
            self.ax[0].plot(self.exact_x,self.exact_h,'.',ms=1,color='black',label='h exact')
            self.ax[1].plot(self.exact_x,self.exact_u,'.',ms=1,color='black',label='u exact')
        
        self.fig.suptitle(f"Time: {self.real_time:.2f} s")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.savefig(f"img/state_{self.t_int}.png")

    ### elliptic non-hydrostatic model ###
    def Tridiagonal(self):
    
        self.A_sub_diag = np.zeros(self.n)
        self.B_diag = np.zeros(self.n)
        self.C_sup_diag = np.zeros(self.n)
        self.D = np.zeros(self.n) #termino independiente
        
        # Auxiliar variables
        phi = 0.5*(np.roll(self.h,-1)+2*np.roll(self.z,-1)-np.roll(self.h,-1)-2*np.roll(self.z,-1)) # mean phi of two walls [i+1]-[i-1]
        phi_edge = (self.h+2*self.z-np.roll(self.h,1)-2*np.roll(self.z,1)) # mean phi of two cells [i]-[i-1]
        h_edge = 0.5*(self.h+np.roll(self.h,1)) # mean h of two cells [i]-[i-1]
        q_edge = 0.5*(self.hu+np.roll(self.hu,1)) # mean q of two cells [i]-[i-1]
        
        # Calculate the coefficients of the tridiagonal matrix
        self.A_sub_diag = (np.roll(phi,1)-2*np.roll(self.h,1))*(phi_edge + 2*h_edge)
        self.B_diag = 16*self.dx**2 + phi_edge*(np.roll(phi,1)+phi+2*np.roll(self.h,1)-2*self.h) + 2*h_edge*(np.roll(phi,1)-phi+4*h_edge)
        self.C_sup_diag = (phi+2*self.h)*(phi_edge - 2*h_edge)
        self.D =-(4*self.dx/self.dt*(self.h*(self.hu-np.roll(self.hu,1))-q_edge*phi_edge+2*self.dx*h_edge*self.w))
            
        self.A_sub_diag[0] = self.A_sub_diag[1] ############################################################################CONTOUR
        self.B_diag[0] = self.B_diag[1]
        self.C_sup_diag[0] = self.C_sup_diag[1]
        self.D[0] = self.D[1]
            

    def TDMA(a,b,c,d):
        n = len(d)
        w= np.zeros(n-1,float)
        g= np.zeros(n, float)
        p = np.zeros(n,float)
        
        w[0] = c[0]/b[0]
        g[0] = d[0]/b[0]

        for i in range(1,n-1):
            w[i] = c[i]/(b[i] - a[i]*w[i-1])
        for i in range(1,n):
            g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
        p[n-1] = g[n-1]
        for i in range(n-1,0,-1):
            p[i-1] = g[i-1] - w[i-1]*p[i]
        return p





















### OUTDATED CODE ###
    # def outdated_entropy_fix(self):
    #     # Entropy fix following Morales-Hernandez, 2014
    #     self.lambda1_hat = np.zeros(self.n)
    #     self.lambda2_hat = np.zeros(self.n)
    #     self.lambda1_bar = np.zeros(self.n)
    #     self.lambda2_bar = np.zeros(self.n)
    #     alpha1_hat = np.zeros(self.n)
    #     alpha2_hat = np.zeros(self.n)
    #     alpha1_bar = np.zeros(self.n)
    #     alpha2_bar = np.zeros(self.n)
    #     beta1_bar = np.zeros(self.n)
    #     beta2_bar = np.zeros(self.n)
    #     self.gamma1_hat = np.zeros(self.n)
    #     self.gamma2_hat = np.zeros(self.n)
    #     self.gamma1_bar = np.zeros(self.n)
    #     self.gamma2_bar = np.zeros(self.n)
        
    #     for i in range(self.n-1):
    #         if self.l1[i]*self.l1[i+1] > 0 and self.l2[i]*self.l2[i+1] > 0:
    #             # default values go to bar variables, 0 otherwise
    #             self.lambda1_bar[i] = self.lambda1[i]
    #             self.lambda2_bar[i] = self.lambda2[i]
    #             self.gamma1_bar[i] = self.gamma1[i]
    #             self.gamma2_bar[i] = self.gamma2[i]
                
    #         else: # Entropy fix will define all pair of values
    #             self.lambda1_bar[i] = self.l1[i]*(self.l1[i+1]-self.lambda1[i])/(self.l1[i+1]-self.l1[i])
    #             self.lambda2_bar[i] = self.l2[i]*(self.l2[i+1]-self.lambda2[i])/(self.l2[i+1]-self.l2[i])
    #             self.lambda1_hat[i] = self.l1[i+1]*(self.l1[i]-self.lambda1[i])/(self.l1[i]-self.l1[i+1])
    #             self.lambda2_hat[i] = self.l2[i+1]*(self.l2[i]-self.lambda2[i])/(self.l2[i]-self.l2[i+1])
    #             alpha1_bar[i] = (self.lambda1_bar[i]*(self.h[i+1]-self.h[i]) - self.u[i+1]*self.h[i+1] + self.u[i]*self.h[i])/(self.width[i]*2*self.c_wall[i])
    #             alpha2_bar[i] = (-self.lambda2_bar[i]*(self.h[i+1]-self.h[i]) + self.u[i+1]*self.h[i+1] - self.u[i]*self.h[i])/(self.width[i]*2*self.c_wall[i])
    #             alpha1_hat[i] = (self.lambda1_hat[i]*(self.h[i+1]-self.h[i]) - self.u[i+1]*self.h[i+1] + self.u[i]*self.h[i])/(self.width[i]*2*self.c_wall[i])
    #             alpha2_hat[i] = (-self.lambda2_hat[i]*(self.h[i+1]-self.h[i]) + self.u[i+1]*self.h[i+1] - self.u[i]*self.h[i])/(self.width[i]*2*self.c_wall[i])
    #             beta1_bar[i] = -(self.gravity*self.h_wall[i]*(self.z[i] - self.z[i+1] - self.S_f[i]*self.dx)*self.dx * self.width[i])/(2*self.c_wall[i])
    #             beta2_bar[i] = -beta1_bar[i]
    #             # beta hat are set to zero
    #             self.gamma1_bar[i] = alpha1_bar[i]-beta1_bar[i]/self.lambda1_bar[i]
    #             self.gamma2_bar[i] = alpha2_bar[i]-beta2_bar[i]/self.lambda2_bar[i]
    #             self.gamma1_hat[i] = alpha1_hat[i] # beta hat are set to zero
    #             self.gamma2_hat[i] = alpha2_hat[i] # beta hat are set to zero
    #     # Last value is the same as the previous one
    #     self.lambda1_bar[-1] = self.lambda1_bar[-2]
    #     self.lambda2_bar[-1] = self.lambda2_bar[-2]
    #     self.lambda1_hat[-1] = self.lambda1_hat[-2]
    #     self.lambda2_hat[-1] = self.lambda2_hat[-2]
    #     self.gamma1_bar[-1] = self.gamma1_bar[-2]
    #     self.gamma2_bar[-1] = self.gamma2_bar[-2]
    #     self.gamma1_hat[-1] = self.gamma1_hat[-2]
    #     self.gamma2_hat[-1] = self.gamma2_hat[-2]