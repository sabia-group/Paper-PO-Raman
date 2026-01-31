#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:24:39 2024

@author: lazzarop
"""

import numpy as np
from ase import io, build
import argparse
import re


abcABC = re.compile(r"CELL[\(\[\{]abcABC[\)\]\}]: ([-+0-9\.Ee ]*)\s*")
angles = np.arange(0, 360, 1)

def grepCell(filename):
    
    with open(filename, 'r') as file:
        file.readline()
        second_line = file.readline()
        comment = second_line.strip()
    strings = [ abcABC.search(comment) ]
    #cells = np.zeros((len(strings),3,3))
    for n,cell in enumerate(strings):
        a, b, c = [float(x) for x in cell.group(1).split()[:3]]
        alpha, beta, gamma = [float(x) for x in cell.group(1).split()[3:6]]
    return [a, b, c, alpha, beta, gamma]

def createStructure(temp, cell):
    modes = np.loadtxt('./' + str(temp) + 'K/modes')
    #If taken from optimized this will be in Bohrs
    equilibriumStructure = io.read('./' + str(temp) + 'K/optimized.xyz', format='extxyz')
    equilibriumStructure.set_cell(cell)
    #eigfreq = np.sqrt(np.loadtxt('./' + str(temp) + 'K/eigval')[3:])*219474.62909

    return modes, equilibriumStructure#, eigfreq

def createRamanT(temp, cutoff):
    polP = np.loadtxt('./' + str(temp) + 'K/pol-plus')[3:cutoff,:]
    polM = np.loadtxt('./' + str(temp) + 'K/pol-minus')[3:cutoff,:]

    polP = np.array([((row.reshape(3, 3) + row.reshape(3, 3).T) / 2).reshape(9) for row in polP])
    polM = np.array([((row.reshape(3, 3) + row.reshape(3, 3).T) / 2).reshape(9) for row in polM])

    return (polP - polM)/(0.0015*2)

def createDisplacementTraj(temp, a, nsteps, nruns):

    displacementTraj = {}
    supercell = [a,a,a]
    scMatrix = np.identity(3)*supercell
    scEquilibriumAtoms = build.make_supercell(equilibriumStructure, scMatrix)
    scEquilibrium = scEquilibriumAtoms.get_positions().flatten()
    
    for r in range(1,nruns+1):
        disptj = []
        traj = io.iread( './' + str(temp) + 'K/' + str(a) + 'x' + str(a) + 'x' + str(a) + '/nve/run' + str(r) + '/nvt.pos_0.xyz', format='extxyz')
        #Assuming trajectory is in Angstroms!
        for i in range(nsteps):
            data = next(traj).get_positions().flatten()/0.52917720859-scEquilibrium
            disptj.append(data)
        displacementTraj[r] = disptj
    return displacementTraj

def coefficientsTraj(displacementTraj, a, nsteps, nruns):
    supercell = [a,a,a]
    interModes = np.linalg.inv(modes)
    interModes = np.tile(interModes, (1,np.prod(supercell)))/np.sqrt(np.prod(supercell))
    coeffTraj = {}
    for r in range(1,nruns+1):
        ctraj = np.empty(nsteps, dtype='object')
        for i in range(nsteps):
            ctraj[i] = interModes@displacementTraj[r][i]
        coeffTraj[r] = ctraj
    return coeffTraj

def RGDOStraj(coeffTraj, ramanTensor, cutoff, nsteps, nruns):
    
    alphaTrajRGDOS = {}
    for r in range(1,nruns+1):
        alphatraj = np.ndarray([nsteps,9])
        for i in range(nsteps):
            alphatraj[i] = np.sum(coeffTraj[r][i][3:cutoff]*ramanTensor.T, axis=1)
        alphaTrajRGDOS[r] = alphatraj
    return alphaTrajRGDOS

def generateCorrelations(polTraj, nsteps, nruns, direction='//'):
    correlationFunction = {}
    for r in range(1,nruns+1):
        pol = polTraj[r]
        correlations = np.empty([len(angles)], dtype=object)
        
        for theta in angles:
            if direction=='//':
                lightDir = np.array([np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta)), 0])
            if direction=='_|_':
                lightDir = np.array([-np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0])
            
            observable = np.zeros([nsteps])
            for j in range(0, nsteps):
                observable[j] = np.dot(np.dot(lightDir, pol[j].reshape(3,3)), lightDir)
            # Subtract avg values s.t. correlation function will go to zero.
            observable -= np.average(observable)
            
            # Compute autocorrelation function for the observable with FT trick.
            ft = np.fft.rfft(observable, axis=0)
            ft_ac = ft*np.conjugate(ft)
            autoCorr = np.fft.irfft(ft_ac)[:int(int(nsteps/2)+1)]/np.average(observable**2)/nsteps

            correlations[theta] = autoCorr 
        correlationFunction[r] = correlations
    return correlationFunction


################


parser = argparse.ArgumentParser()
parser.add_argument('--nruns', dest='nruns', type=int, required=True)
parser.add_argument('--nsteps', dest='nsteps', type=int, required=True)
parser.add_argument('--temp', dest='temp', nargs='+', type=int, required=True)
parser.add_argument('--dim', dest='sc', nargs=1, type=int, required=True)
parser.add_argument('--out', dest='output', type=str, default='output')
parser.add_argument('--nmodes', dest='cutoff', type=int, required=True)

args = parser.parse_args()


nruns=args.nruns
nsteps=args.nsteps

corrSupercell = {}
for s in args.sc:
    corr = {}
    for temp in args.temp:
        
        modes, equilibriumStructure = createStructure(temp, grepCell( './' + str(temp) + 'K/optimized.xyz'))
        ramanTensor = createRamanT(temp, args.cutoff)             
        ramanTensor = ramanTensor 
        dtraj = createDisplacementTraj(temp, s, nsteps=nsteps, nruns=nruns)
        ctraj = coefficientsTraj(dtraj, s, nsteps=nsteps, nruns=nruns)
        del(dtraj)
        alpha = RGDOStraj(ctraj, ramanTensor, args.cutoff, nsteps=nsteps, nruns=nruns)
        del(ctraj)
        corr[temp] = generateCorrelations(alpha, nsteps=nsteps, nruns=nruns, direction='//')
        del(alpha)
        
    corrSupercell[s] = corr
    np.save(args.output + '.npy', corrSupercell)
