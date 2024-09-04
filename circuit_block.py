from qiskit import QuantumCircuit
import numpy as np


def A(theta, phi):
    qc=QuantumCircuit(2)
    qc.cx(0,1)
    qc.rz(phi+np.pi, 0).inverse()
    qc.ry(theta+np.pi/2, 0).inverse()
    qc.cx(1,0)
    qc.ry(theta+np.pi/2, 0)
    qc.rz(phi+np.pi, 0)
    qc.cx(0,1)
    return qc

