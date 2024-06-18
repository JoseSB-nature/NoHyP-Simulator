from lib.canal import *
import os

if __name__ == '__main__':
    
    # Clear files
    os.system('rm -rf img/*')
  
    river = Canal()

    river.calc_vectors()
    river.temporal_loop(mode='flux') # Can be 'flux' or 'cont' for fluxes or contributions scheme 
    
    # river.prog_bar.next()
    river.prog_bar.finish()