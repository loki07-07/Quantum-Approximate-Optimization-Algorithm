# Quantum-Approximate-Optimization-Algorithm

# QAOA MaxCut Problem Solver

A quantum algorithm implementation for solving the Maximum Cut (MaxCut) problem using the Quantum Approximate Optimization Algorithm (QAOA) on IBM's Qiskit framework.

## Overview

The Maximum Cut problem is a classic NP-hard problem in combinatorial optimization. Given an undirected graph, MaxCut aims to partition the vertices into two sets such that the number of edges between the sets is maximized. This implementation uses QAOA, a variational quantum algorithm suitable for near-term quantum computers (NISQ devices), to find approximate solutions.

## Features

- **QAOA Circuit Construction**: Builds parameterized quantum circuits with configurable depth
- **Classical Optimization**: Uses COBYLA optimizer to find optimal QAOA parameters
- **Multiple Simulator Support**: Automatically selects the best available simulator (Qiskit Aer, Qiskit Primitives, or random fallback)
- **Measurement Analysis**: Provides detailed statistics on measured quantum states and their cut sizes
- **Approximation Ratio Calculation**: Evaluates solution quality against the theoretical maximum cut

## Requirements

```
networkx
numpy
scipy
matplotlib
qiskit
qiskit-aer  (optional but recommended)
```

Install dependencies with:
```bash
pip install networkx numpy scipy matplotlib qiskit qiskit-aer
```

## Usage

### Basic Usage

Run the script directly to solve MaxCut on the default 4-node square graph:

```bash
python qaoa_maxcut.py
```

### Customizing the Graph

Modify the graph definition in the `main()` function:

```python
import networkx as nx
from scipy.optimize import minimize
from qaoa_maxcut import get_expectation, create_qaoa_circuit, simulate_circuit, compute_expectation, maxcut_obj

# Create your own graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])

print(f"Graph Edges: {list(G.edges())}")
print(f"Maximum Possible Cut: {len(G.edges())}")

# Run optimization
p = 1
expectation = get_expectation(G, p=p)
init_params = [1.0, 1.0]

res = minimize(expectation, init_params, method='COBYLA', 
               options={'maxiter': 50})

print(f"Best Expectation: {res.fun}")
print(f"Optimal Parameters: {res.x}")
```

### Adjusting QAOA Depth

Change the `p` parameter (circuit depth) to improve solution quality:

```python
p = 2  # Increase depth for better approximations
expectation = get_expectation(G, p=p)

# Need 2p parameters for depth p
init_params = [1.0] * (2 * p)

res = minimize(expectation, init_params, method='COBYLA', 
               options={'maxiter': 100, 'tol': 1e-4})

print(f"Optimal beta: {res.x[:p]}")
print(f"Optimal gamma: {res.x[p:]}")
```

### Running with Custom Parameters

```python
from qaoa_maxcut import create_qaoa_circuit, simulate_circuit, compute_expectation

# Use pre-defined parameters without optimization
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

optimal_theta = [0.5, 0.8]  # Your optimal parameters
qc = create_qaoa_circuit(G, optimal_theta)

# Simulate
counts = simulate_circuit(qc, shots=5000)

# Analyze results
expectation = compute_expectation(counts, G)
print(f"Expectation Value: {expectation}")

# Find best state
best_state = max(counts, key=counts.get)
print(f"Best State: {best_state}")
```

### Programmatic Usage

```python
import networkx as nx
from qaoa_maxcut import get_expectation, create_qaoa_circuit, simulate_circuit
from scipy.optimize import minimize

def solve_maxcut(edges, p=1, shots=1024):
    """Solve MaxCut on a graph defined by edges."""
    G = nx.Graph()
    G.add_edges_from(edges)
    
    print(f"Solving MaxCut with p={p}")
    expectation = get_expectation(G, p=p)
    
    init_params = [1.0] * (2 * p)
    res = minimize(expectation, init_params, method='COBYLA')
    
    # Get final circuit with optimal params
    qc = create_qaoa_circuit(G, res.x)
    counts = simulate_circuit(qc, shots=shots)
    
    return {
        'graph': G,
        'result': res,
        'counts': counts,
        'optimal_params': res.x
    }

# Example
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
solution = solve_maxcut(edges, p=2)
print(f"Best cut found: {-solution['result'].fun}")
```

## Key Functions

### `create_qaoa_circuit(graph, theta)`
Constructs a QAOA circuit with the given parameters.
- **Parameters**: `graph` (NetworkX Graph), `theta` (list of parameters)
- **Returns**: Qiskit QuantumCircuit

### `maxcut_obj(x, graph)`
Calculates the cut size for a given bitstring solution.
- **Parameters**: `x` (bitstring), `graph` (NetworkX Graph)
- **Returns**: Negative cut size (for minimization)

### `compute_expectation(counts, graph)`
Computes the expectation value from measurement statistics.
- **Parameters**: `counts` (dict of bitstrings and counts), `graph` (NetworkX Graph)
- **Returns**: Average expectation value

### `simulate_circuit(qc, shots=1024)`
Executes the circuit on available simulators with fallback support.
- **Parameters**: `qc` (QuantumCircuit), `shots` (number of measurements)
- **Returns**: Dictionary of measurement results

### `get_expectation(graph, p=1)`
Returns a function that evaluates QAOA performance for given parameters.
- **Parameters**: `graph` (NetworkX Graph), `p` (circuit depth)
- **Returns**: Callable function for optimization

## Output Example

```
======================================================================
QAOA MaxCut Problem Solver
======================================================================

Graph Nodes: [0, 1, 2, 3]
Graph Edges: [(0, 1), (1, 2), (2, 3), (3, 0)]
Maximum Possible Cut Size: 4

Starting QAOA optimization with p=1...

======================================================================
Optimization Result:
======================================================================
Success: True
Best Expectation Value: -3.123456
Optimal Parameters: beta=0.654321, gamma=0.789012
Function Evaluations: 45

======================================================================
Final Results with Optimal Parameters (5000 shots):
======================================================================

State      Count   Probability  Cut Size  
---------------------------------------------
|0101>     1250    0.2500       4         
|1010>     1240    0.2480       4         
|0110>     380     0.0760       3         
|1001>     380     0.0760       3         

Expectation Value: -3.123456
Best State Found: |0101> with Cut Size: 4
Maximum Possible Cut Size: 4
Approximation Ratio: 4/4 = 100.0%
======================================================================
```

## How QAOA Works

1. **Initialization**: All qubits are placed in superposition with Hadamard gates
2. **Cost Hamiltonian**: Encodes the MaxCut problem via controlled phase rotations on edges
3. **Mixer Hamiltonian**: Applies X-rotations to explore the solution space
4. **Measurement**: Collapses the quantum state to classical bitstrings
5. **Classical Optimization**: Adjusts circuit parameters to maximize the expected cut size

The depth `p` determines how many times the cost and mixer unitaries are applied. Higher values generally provide better approximations but require more quantum resources.

## Optimization Algorithm

This implementation uses the COBYLA (Constrained Optimization BY Linear Approximation) method, a classical gradient-free optimization algorithm well-suited for noisy quantum simulations.

## Simulator Priority

The code automatically selects simulators in this order:
1. **Qiskit Aer** - Most performant (compiled C++ backend)
2. **Qiskit Primitives** - Built-in for Qiskit 1.0+
3. **Random Fallback** - Generates random bitstrings if no simulator available

## Limitations & Notes

- The implementation currently supports small graphs (typically up to 15-20 qubits on simulators)
- Results are stochastic due to quantum measurement randomness
- On real quantum hardware, noise will affect solution quality
- Bitstring ordering may vary between simulators; the code includes correction logic

## Future Enhancements

- Support for real quantum hardware (IBM Quantum devices)
- Warm-starting with classical heuristics
- Automatic parameter initialization
- Visualization of circuit and results
- Support for larger graphs with error mitigation

## References

- Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm"
- Qiskit Documentation: https://qiskit.org/documentation/
- NetworkX Documentation: https://networkx.org/

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
