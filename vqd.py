import pennylane as qml
import pennylane.numpy as np
import jax
import jax.numpy as jnp
import torch

import circuit


def three_spins_forward_state(params, delta_time, n=1):
    circuit.three_spins_circuit(params)
    qml.ApproxTimeEvolution(circuit.hamiltonian(), delta_time, n)
    return qml.state()


def three_spins_current_state(params):
    circuit.three_spins_circuit(params)
    return qml.state()


def three_spins_observables(params):
    circuit.three_spins_circuit(params)
    retvals = [qml.expval(qml.PauliX(i)) for i in range(3)]
    retvals += [qml.expval(qml.PauliZ(i)) for i in range(3)]
    return retvals


def three_spins_observables_variance(params):
    circuit.three_spins_circuit(params)
    retvals = [qml.var(qml.PauliX(i)) for i in range(3)]
    retvals += [qml.var(qml.PauliZ(i)) for i in range(3)]
    return retvals


def three_spins_observables_samples_sigx(params):
    circuit.three_spins_circuit(params)
    retvals = [qml.sample(qml.PauliX(i)) for i in range(3)]
    return retvals


def three_spins_observables_samples_sigz(params):
    circuit.three_spins_circuit(params)
    retvals = [qml.sample(qml.PauliZ(i)) for i in range(3)]
    return retvals


class VQD:
    """

    """
    def __init__(self, interface, total_time, delta_time,
                 max_iterations, cost_threshold, n_qubits=3, shots=800,
                 cost_function='fidelity', optimization_step_size=0.1,
                 device_type='default.qubit', predefined_state_device=None, predefined_observable_device=None):
        """

        If predefined_state_device or predefined_observable_device are defined, they are expected to be
        pennylane.device objects
        """
        assert n_qubits == 3

        self.total_time = total_time
        self.delta_time = delta_time
        self.max_iterations = max_iterations
        self.cost_threshold = cost_threshold
        self.shots = shots
        self.optimization_step_size = optimization_step_size
        self.reset_params = None
        self.state_device = None
        self.sample_device = None
        self.device_type = device_type
        self.interface = None
        self.predefined_state_device = predefined_state_device
        self.predefined_observable_device = predefined_observable_device

        self.qnode_three_spins_forward_state = None
        self.qnode_three_spins_current_state = None
        self.qnode_three_spins_observables = None
        self.current_params = None
        self.previous_params = None
        self.previous_state = None
        self.set_interface(interface)

        if cost_function == 'fidelity':
            self.cost_function = self._qml_fidelity_cost_function
        else:
            assert NotImplementedError(cost_function)

    def set_interface(self, interface):
        """

        :param interface:
        :return:
        """
        self.interface = interface

        if interface == 'autograd':
            # import pennylane.numpy as np

            def reset_params():
                return 0.0 * np.ones(18)

            self.reset_params = reset_params

        elif interface == 'jax':
            # import jax
            # import jax.numpy as jnp

            def reset_params():
                return jnp.array([0.0] * 18)

            self.reset_params = reset_params

        elif interface == 'torch':
            # import torch
            def reset_params():
                return torch.tensor([0.0] * 18, requires_grad=True)

            self.reset_params = reset_params

        else:
            raise NotImplementedError(f'interface {interface} not supported')

        if self.predefined_state_device:
            self.state_device = self.predefined_state_device
        else:
            # dev = qml.device("default.qubit.jax", wires=range(3)) raises TypeError with default.qubit.jax
            self.state_device = qml.device(self.device_type, wires=range(3))

        if self.predefined_observable_device:
            self.sample_device = self.predefined_observable_device
        else:
            self.sample_device = qml.device(self.device_type, wires=range(3), shots=self.shots)

        self.qnode_three_spins_forward_state = qml.QNode(three_spins_forward_state, self.state_device, interface=interface)
        self.qnode_three_spins_current_state = qml.QNode(three_spins_current_state, self.state_device, interface=interface)
        self.qnode_three_spins_observables = qml.QNode(three_spins_observables, self.sample_device, interface=interface)
        self.qnode_three_spins_observables_variance = qml.QNode(three_spins_observables_variance, self.sample_device, interface=interface)
        self.qnode_three_spins_observables_sample_sigx = qml.QNode(three_spins_observables_samples_sigx, self.sample_device, interface=interface)
        self.qnode_three_spins_observables_sample_sigz = qml.QNode(three_spins_observables_samples_sigz, self.sample_device, interface=interface)

    def _qml_fidelity_cost_function(self, params):
        self.forward_state = self.qnode_three_spins_forward_state(params, self.delta_time)

        fidelity = qml.math.fidelity(self.previous_state, self.forward_state)
        return 1 - fidelity

    def run_optimization(self, compute_observables=True):

        self.current_params = self.reset_params()
        self.previous_params = self.reset_params()
        self.previous_state = self.qnode_three_spins_current_state(self.previous_params)

        opt = qml.GradientDescentOptimizer(stepsize=self.optimization_step_size)

        final_costs_v_time = []
        full_costs_v_time = []
        final_params_v_time = []
        full_params_v_time = []
        failed_to_converge_times = {}
        number_of_iterations_to_converge = []
        observables = []
        variances = []
        time = []

        try:
            for current_time in np.arange(0, self.total_time + self.delta_time, self.delta_time):
                print(current_time)

                recorded_params = [self.current_params]
                recorded_costs = [self.cost_function(self.current_params)]

                for n in range(self.max_iterations):
                    self.current_params, prev_cost = opt.step_and_cost(self.cost_function, self.current_params)

                    recorded_costs.append(self.cost_function(self.current_params))
                    recorded_params.append(self.current_params)

                    if recorded_costs[-1] <= self.cost_threshold:
                        break

                if recorded_costs[-1] > self.cost_threshold:
                    failed_to_converge_times[current_time] = recorded_costs[-1]

                # prepare for next time step
                self.previous_state = self.qnode_three_spins_current_state(recorded_params[-1])

                # record results
                time.append(current_time)
                if compute_observables:
                    observables.append(self.qnode_three_spins_observables(self.current_params, shots=self.shots))
                    variances.append(self.qnode_three_spins_observables_variance(self.current_params, shots=self.shots))

                final_costs_v_time.append(recorded_costs[-1])
                full_costs_v_time.append(recorded_costs)
                final_params_v_time.append(recorded_params[-1])
                full_params_v_time.append(recorded_params)
                number_of_iterations_to_converge.append(n)

        except Exception as e:
            print(e)
            raise e

        finally:

            output = dict()
            output["final_costs_v_time"] = final_costs_v_time
            output["full_costs_v_time"] = full_costs_v_time
            output["final_params_v_time"] = final_params_v_time
            output["full_params_v_time"] = full_params_v_time
            output["failed_to_converge_times"] = failed_to_converge_times
            output["number_of_iterations_to_converge"] = number_of_iterations_to_converge
            output["observables"] = observables
            output["variances"] = variances
            output["time"] = time

            self.last_run_output = output

            return output
