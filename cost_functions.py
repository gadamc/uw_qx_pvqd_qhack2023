import pennylane as qml
import pennylane.numpy as np


def fidelity(state0, state1, dt):
    return (1 - qml.math.fidelity(state0, state1)) / dt ** 2


### In development below

def projector_zero(n_qubits):
    zero_state = np.zeros(2 ** n_qubits)
    zero_state[0] = 1

    return np.outer(zero_state, zero_state)


def jth_projector(n_qubits, j):
    jth_state = np.zeros(n_qubits)
    jth_state[j] = 1

    eye_bar = np.eye(n_qubits)
    eye_bar[j, j] = 0

    return np.tensordot(np.outer(jth_state, jth_state), eye_bar, axis=0)


def local_fidelity(state0, state1):
    n_qubits = 3  # how can we get this value from the state0 object?

    for j in range(n_qubits):
        jth_state = np.zeros(n_qubits)
        jth_state[j] = 1

        eye_bar = np.eye(n_qubits)
        eye_bar[j, j] = 0
        jth_prod += np.tensordot(np.outer(jth_state, jth_state) / n_qubits, eye_bar, axis=0)

    np.eye(n_qubits) - np.tensordot(np.outer(jth_state, jth_state) / n_qubits, eye_bar, axis=0)


def global_op(n_qubits):
    return np.eye(2 ** n_qubits) - projector_zero(n_qubits)


### original qiskit implementation

def projector__zero(n_qubits):
    # This function create the global projector |00...0><00...0|
    from qiskit.opflow import Z, I

    prj_list = [0.5 * (I + Z) for i in range(n_qubits)]
    prj = prj_list[0]

    for a in range(1, len(prj_list)):
        prj = prj ^ prj_list[a]

    return prj


def projector_zero_local(n_qubits):
    # This function creates the local version of the cost function
    # proposed by Cerezo et al: https://www.nature.com/articles/s41467-021-21728-w
    from qiskit.opflow import Z, I

    tot_prj = 0

    for k in range(n_qubits):
        prj_list = [I for i in range(n_qubits)]
        prj_list[k] = 0.5 * (I + Z)
        prj = prj_list[0]

        for a in range(1, len(prj_list)):
            prj = prj ^ prj_list[a]

        # print(prj)

        tot_prj += prj

    tot_prj = (1 / n_qubits) * tot_prj

    return tot_prj
