import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=range(3))


def hamiltonian(coupling_strength=0.25, field_strength=1):
    coeffs = [-field_strength]*3
    coeffs += [coupling_strength]*6

    obs = [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2),
           qml.PauliX(0) @ qml.PauliX(1),
           qml.PauliY(0) @ qml.PauliY(1),
           qml.PauliZ(0) @ qml.PauliZ(1),
           qml.PauliX(0) @ qml.PauliX(1),
           qml.PauliY(0) @ qml.PauliY(1),
           qml.PauliZ(0) @ qml.PauliZ(1),
           ]
    return qml.Hamiltonian(coeffs, obs)


def three_spins_circuit(params):
    """
    Constructs the parameterized circuit that transforms the |000> state to our time-evolved state.

    NB -- this is an explict declaration of the circuit rather for n = 3 spin components and a circuit
    depth of 3. Future versions of this code could allow for arbitrary values
    :param params: array-like object of size 18; (depth+1)*n + depth*(n-1)
    :return:
    """
    qml.RX(params[0], wires=0)
    qml.RX(params[1], wires=1)
    qml.RX(params[2], wires=2)

    qml.IsingZZ(params[3], wires=[0,1])
    qml.IsingZZ(params[4], wires=[1,2])

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


@qml.qnode(dev)
def three_spins(time, n):
    """

    """

    params = np.random.random(18)
    three_spins_circuit(params)
    qml.ApproxTimeEvolution(hamiltonian(), time, n)
    return qml.state()
