# Implementation of the Projected Variational Quantum Dynamics Algorithm

We implement the 
[projected Variational Quantum Dynamics algorithm](https://doi.org/10.22331/q-2021-07-28-512) 
for multiple coupled spins in a transverse uniform magnetic field using Pennylane.

We plan to test and study the fidelity (our cost function) at each time step, as well as Pauli spin 
observables and compare our results to the published outcomes. 

Our conclusions are found in [quantumx_uw_team_writeup.pdf](quantumx_uw_team_writeup.pdf).

Main sections of the code are circuit.py, vqd.py and test_vqd.ipynb. 



#### Notes

Pytorch does not yet support Python 3.11. Using Python 3.10.9.

After installing pennylane, I am trying a number of different ML platforms because autograd didn't work
initially. I first installed JAX, but was getting errors. I then installed PyTorch into the same environment. 
However, this installation order caused an error:

```
ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant'
```

Previously, I installed pennylane, then pytorch and did not receive this error. So, once we decide upon the 
correct ML tool, we probably won't run into this problem. But, if you are following along and installed 
pennylane, then JAX, then PyTorch, to work around this error I then installed chardet `pip install chardet` and
the import error above goes away. 

Installing ipython, jupyterlab and matplotlib (conda install).

##### 5pm Friday

Using 'autograd' I have something that seems to be working!  Use the `test_circuit.ipynb` notebook. 