import ctypes

#Load the C shared object file 
lib = ctypes.CDLL('./Quplexity/ARM/gates.so')
##

#Import Hadamard Gate f()
lib.gills_hadamard.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
##

