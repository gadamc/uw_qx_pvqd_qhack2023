import sys
import pennylane as qml

def hamiltonian(coupling_strength=0.25, field_strength=1):
    coefficients = [-field_strength]*3
    coefficients += [coupling_strength]*2

    obs = [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2),
           qml.PauliZ(0) @ qml.PauliZ(1),
           qml.PauliZ(1) @ qml.PauliZ(2),
           ]
    return qml.Hamiltonian(coefficients, obs)


def three_spins_circuit(params, n_spins = 3, depth= 3):
    """
    Constructs the parameterized circuit that transforms the |000> state to our time-evolved state.

    NB -- this is an explict declaration of the circuit for n = 3 spin components and a circuit
    depth of 3. Future versions of this code could allow for arbitrary values
    :param params: array-like object of size 18; (depth+1)*n + depth*(n-1)
    :param n_spins: number of particles with spin state
    :param depth: depth of the circuit
    :return: None
    """
    qml.RX(params[0], wires=0)
    qml.RX(params[1], wires=1)
    qml.RX(params[2], wires=2)

    qml.IsingZZ(params[3], wires=[0, 1])
    qml.IsingZZ(params[4], wires=[1, 2])

    qml.RY(params[5], wires=0)
    qml.RY(params[6], wires=1)
    qml.RY(params[7], wires=2)

    qml.IsingZZ(params[8], wires=[0, 1])
    qml.IsingZZ(params[9], wires=[1, 2])

    qml.RX(params[10], wires=0)
    qml.RX(params[11], wires=1)
    qml.RX(params[12], wires=2)

    qml.IsingZZ(params[13], wires=[0, 1])
    qml.IsingZZ(params[14], wires=[1, 2])

    qml.RY(params[15], wires=0)
    qml.RY(params[16], wires=1)
    qml.RY(params[17], wires=2)

