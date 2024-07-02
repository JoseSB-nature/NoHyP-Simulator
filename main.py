from lib.canal import *
import os

if __name__ == '__main__':
    
    # Clear files
    os.system('rm -rf img/*')
  
    # river = Canal()

    # river.calc_vectors()
    # river.temporal_loop(mode='wave') # Can be 'flux' or 'wave' for fluxes or contributions scheme 
    
    river = Canal()

    river.temporal_loop(mode='flux') # Can be 'flux' or 'wave' for fluxes or contributions scheme     
    
    # river.prog_bar.next()
    river.prog_bar.finish()
    
    river.save_config()