import sys
import pennylane as qml

def hamiltonian(coupling_strength=0.25, field_strength=1):
    coefficients = [-field_strength]*3
    coefficients += [coupling_strength]*6

    obs = [qml.PauliX(0), qml.PauliX(1), qml.PauliX(2),
           qml.PauliX(0) @ qml.PauliX(1),
           qml.PauliY(0) @ qml.PauliY(1),
           qml.PauliZ(0) @ qml.PauliZ(1),
           qml.PauliX(1) @ qml.PauliX(2),
           qml.PauliY(1) @ qml.PauliY(2),
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

def run(interface):

    if interface == 'autograd':
        import pennylane.numpy as np
        dev = qml.device("default.qubit", wires=range(3))
        params = 0.05 * np.ones(18)

    elif interface == 'jax':
        import jax
        import jax.numpy as jnp
        # next line raises TypeError with default.qubit.jax
        # dev = qml.device("default.qubit.jax", wires=range(3))
        dev = qml.device("default.qubit", wires=range(3))
        params = jnp.array([0.05] * 18)
    elif interface == 'torch':
        import torch
        dev = qml.device("default.qubit", wires=range(3))
        params = torch.tensor([0.05] * 18, requires_grad=True)
    else:
        return -1

    @qml.qnode(dev, interface=interface)
    def three_spins_forward(params, delta_time, n=1):
        """
        params - array / tensor of size 18

        """
        three_spins_circuit(params)
        # print(pre_time_evolution_state)

        # or something like --> qml.exp(hamiltonian(), delta_time)?
        # all I really want to do is U(delta_time) = exp(-i * hamiltonian() *delta_time)
        # That is... I only want to take one trotter step for this small delta_time.
        # Does ApproxTimeEvolution(hamiltonion(), delta_time, n = 1) accomplish that?
        # Although...it can't hurt to have a n > 1, as we still are only performing U(deltat_time)
        qml.ApproxTimeEvolution(hamiltonian(), delta_time, n)
        #print(qml.state())

        # Seems like we can't use this for computing gradients -- raises exceptions
        #fidelity = qml.math.fidelity(pre_time_evolution_state, qml.state())
        # return fidelity
        # looking at this --> https://strawberryfields.ai/photonics/demos/run_state_learner.html

        # this works --
        # return qml.expval(qml.PauliZ(0))

        # this does not work -- error is RX must occur prior to measurement
        # Obviously I cannot read the quantum state, then perform an operation!
        # pre_time_evolution_state = qml.state()
        # qml.RX(params[0], wires=0)
        # return [pre_time_evolution_state, qml.state()]
        #
        #  same error here
        # qml.RX(params[0], wires=0)
        # return qml.state()

        return qml.state()

    @qml.qnode(dev, interface=interface)
    def three_spins_current(params):
        three_spins_circuit(params)
        return qml.state()
    dt = 0.1
    original_state = three_spins_current(params)
    forward_state = three_spins_forward(params + 0.01, dt)

    fidelity = qml.math.fidelity(original_state, forward_state)

    return fidelity

if __name__ == '__main__':
    print(sys.argv)
    print(run(sys.argv[1]))

# @qml.qnode(dev, interface=interface)
# def three_spins(params, delta_time, n=1):
#     """
#     params - array / tensor of size 18
#
#     """
#     three_spins_circuit(params)
#     #pre_time_evolution_state = qml.state()
#     #print(pre_time_evolution_state)
#     qml.ApproxTimeEvolution(hamiltonian(), delta_time, n)  # or should we just do qml.exp(hamiltonian(), delta_time)?
#     #print(qml.state())
#     # fidelity = qml.math.fidelity(pre_time_evolution_state, qml.state())
#     return qml.expval(qml.PauliZ(0))

# if __name__ == '__main__':
#     if interface == 'jax':
#         params = jnp.array(0.05 * [1]*18)
#     elif interface == 'autograd':
#         params = 0.05 * np.ones(18)
#
#     cost_threshold = 0.9999
#     max_iterations = 50
#
#     delta_time = 0.1
#     total_time = 2.0
#
#     for time_step_index in range(1, int(total_time / delta_time)):
#         current_time = time_step_index * delta_time
#         fidelity = 0
#         num_iterations = 0
#
#         while fidelity < cost_threshold and num_iterations < max_iterations:
#             fidelity = three_spins(params, delta_time)
#
#
#
#             cost = (1 - fidelity)   # / delta_time**2
#             num_iterations += 1




