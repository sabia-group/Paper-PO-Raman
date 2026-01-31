from ase import units
from ase.io import read, write
import numpy as np
import time

from mace.calculators import MACECalculator


#from ase.calculators.aims import Aims

# from ipi.interfaces.clients import ClientASE
from ase.calculators.socketio import SocketClient
from ase.io import read

# setting up MACE calculator
print("------ Imports done ------", flush = True)
calculator = MACECalculator(model_paths='/u/lazerpo/poRaman/anthra-MACE/anthra_float64_swa.model', device='cuda', default_dtype='float64')
init_conf = read('anthra100K.cif', format='cif')
init_conf.set_calculator(calculator)

# Create Client ############################
# inet
port = 23423 
host = "unixsocketipi" 
#client = SocketClient(host=host, port=port)
client = SocketClient(unixsocket=host)

# 'unix'
# client = SocketClient(unixsocket=host)
print("-------- Starting the run -------", flush = True)
client.run(init_conf, use_stress = True)
print("-------- This is the end --------", flush = True)
