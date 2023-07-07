# Importing standard Qiskit libraries
from qiskit import (
    QuantumCircuit,
    execute,
    Aer,
    IBMQ,
    QuantumRegister,
    ClassicalRegister,
)
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.circuit.library import QFT
from numpy import pi
from qiskit.quantum_info import Statevector
from matplotlib import pyplot as plt

one_step_circuit = QuantumCircuit(6, name=" ONE STEP")
# Coin operator
one_step_circuit.h([4, 5])
one_step_circuit.z([4, 5])
one_step_circuit.cz(4, 5)
one_step_circuit.h([4, 5])
one_step_circuit.draw()


# Shift operator dunction for 4d-hypercube
def shift_operator(circuit):
    for i in range(4):
        circuit.x(4)
        if i % 2 == 0:
            circuit.x(5)
        circuit.ccx(4, 5, i)


shift_operator(one_step_circuit)
one_step_gate = one_step_circuit.to_instruction()
one_step_circuit.draw()

# one_step_inv = one_step_circuit.inverse().draw()
# one_step_circuit.inverse().draw()

# print(one_step_circuit.inverse().draw())

inv_cont_one_step = one_step_circuit.inverse().control()
inv_cont_one_step_gate = inv_cont_one_step.to_instruction()
cont_one_step = one_step_circuit.control()
cont_one_step_gate = cont_one_step.to_instruction()

inv_qft_gate = QFT(4, inverse=True).to_instruction()
qft_gate = QFT(4, inverse=False).to_instruction()

QFT(4, inverse=True).decompose().draw("mpl")
# plt.show()

phase_circuit = QuantumCircuit(6, name=" phase oracle ")
# Mark 1011
phase_circuit.x(2)
phase_circuit.h(3)
phase_circuit.mct([0, 1, 2], 3)
phase_circuit.h(3)
phase_circuit.x(2)
# Mark 1111
phase_circuit.h(3)
phase_circuit.mct([0, 1, 2], 3)
phase_circuit.h(3)
phase_oracle_gate = phase_circuit.to_instruction()
# phase oracle circuit
phase_oracle_circuit = QuantumCircuit(11, name=" PHASE ORACLE CIRCUIT ")
phase_oracle_circuit.append(phase_oracle_gate, [4, 5, 6, 7, 8, 9])
# phase_circuit.draw()

mark_auxiliary_circuit = QuantumCircuit(5, name=" mark auxiliary ")
mark_auxiliary_circuit.x([0, 1, 2, 3, 4])
mark_auxiliary_circuit.mct([0, 1, 2, 3], 4)
mark_auxiliary_circuit.z(4)
mark_auxiliary_circuit.mct([0, 1, 2, 3], 4)
mark_auxiliary_circuit.x([0, 1, 2, 3, 4])

mark_auxiliary_gate = mark_auxiliary_circuit.to_instruction()
mark_auxiliary_circuit.draw()

phase_estimation_circuit = QuantumCircuit(11, name=" name estimation ")
phase_estimation_circuit.h([0, 1, 2, 3])
for i in range(4):
    stop = 2**i
    for j in range(0, stop):
        phase_estimation_circuit.append(cont_one_step, [i, 4, 5, 6, 7, 8, 9])

# Inverse Fourier transform
phase_estimation_circuit.append(inv_qft_gate, [0, 1, 2, 3])

# Mark all angles theta that are not 0 wth an auxillary qubit
phase_estimation_circuit.append(mark_auxiliary_gate, [0, 1, 2, 3, 10])

# Reverse phase estimation
phase_estimation_circuit.append(qft_gate, [0, 1, 2, 3])

for i in range(3, -1, -1):
    stop = 2**i
    for j in range(0, stop):
        phase_estimation_circuit.append(inv_cont_one_step, [i, 4, 5, 6, 7, 8, 9])
phase_estimation_circuit.barrier(range(0, 10))
phase_estimation_circuit.h([0, 1, 2, 3])

# Make phase estimation gate
phase_estimation_gate = phase_estimation_circuit.to_instruction()
phase_estimation_circuit.draw()
print(phase_estimation_circuit)
