#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:25:59 2024

@author: lazzarop
"""

import numpy as np
from ase import io, build
import copy
import sys

x0 = io.read('optimized.xyz', format='extxyz')

#Change this according to cell size, if i-PI format, copy from the comment of optmized.xyz
x0.set_cell([ 17.55121 ,   11.31984 ,   15.89770 ,   90.00000 ,  102.52600  ,  90.00000 ])

modes = np.loadtxt('modes')
displacement = sys.argv[1]

for vec in modes.T:
        newatoms=copy.deepcopy(x0)
        dispvec = vec.reshape(-1,3)
        
        # plus
        newatoms=copy.deepcopy(x0)
        newatoms.positions=newatoms.positions+(float(displacement)*dispvec)
        filename= str(displacement) + "plus.xyz"
        
        io.write(filename, newatoms, format='extxyz', append=True)
        params = np.append(x0.cell.lengths(), x0.cell.angles())

        # minus 
        newatoms=copy.deepcopy(x0)
        newatoms.positions=newatoms.positions-(float(displacement)*dispvec)
        filename=str(displacement) + "minus.xyz"
        
        io.write(filename, newatoms, format='extxyz', append=True)
        params = np.append(x0.cell.lengths(), x0.cell.angles())
