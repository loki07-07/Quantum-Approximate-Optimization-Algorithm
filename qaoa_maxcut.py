import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def create_qaoa_circuit(graph, theta):
    """
    Creates a QAOA circuit for the MaxCut problem.
    """
    beta = theta[:len(theta)//2]
    gamma = theta[len(theta)//2:]
    n_qubits = len(graph.nodes)
    
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # Initialization: Apply Hadamard to all qubits
    for i in range(n_qubits):
        circuit.h(qr[i])
    
    # Number of layers (p)
    p = len(beta)
    
    for i in range(p):
        # Problem Unitary (Cost Hamiltonian)
        for u, v in graph.edges():
            circuit.cx(qr[u], qr[v])
            circuit.rz(2 * gamma[i], qr[v])
            circuit.cx(qr[u], qr[v])
            
        # Mixer Unitary
        for node in graph.nodes():
            circuit.rx(2 * beta[i], qr[node])
    
    # Measure all qubits
    for i in range(n_qubits):
        circuit.measure(qr[i], cr[i])
    
    return circuit

def maxcut_obj(x, graph):
    """
    Calculate the MaxCut objective value for a given bitstring.
    Returns negative of cut size (for minimization).
    """
    cut_size = 0
    for i, j in graph.edges():
        # Convert string to integer if needed
        if isinstance(x, str):
            xi = int(x[i])
            xj = int(x[j])
        else:
            xi = x[i]
            xj = x[j]
        
        if xi != xj:
            cut_size += 1
    return -cut_size

def compute_expectation(counts, graph):
    """
    Computes the expectation value based on measurement counts.
    """
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = maxcut_obj(bitstring, graph)
        avg += obj * count
        sum_count += count
    
    if sum_count == 0:
        return 0
    return avg / sum_count

def get_simulator():
    """Get the best available simulator."""
    try:
        from qiskit_aer import AerSimulator
        print("Using Qiskit-Aer simulator")
        return AerSimulator()
    except ImportError:
        print("Qiskit Aer not found. Using Qiskit Basic Simulator.")
        return None

def simulate_circuit(qc, shots=1024):
    """
    Simulate a circuit using available backends.
    Returns a dictionary of counts.
    """
    # 1. Try Qiskit Aer (Preferred)
    try:
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
        # Transpile for the backend
        from qiskit import transpile
        qc_transpiled = transpile(qc, simulator)
        job = simulator.run(qc_transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        return counts
    except ImportError:
        pass
    
    # 2. Try Qiskit StatevectorSampler (Built-in for Qiskit 1.0+)
    try:
        from qiskit.primitives import StatevectorSampler
        # StatevectorSampler returns quasi-dists in a PubResult
        sampler = StatevectorSampler()
        job = sampler.run([qc], shots=shots) 
        result = job.result()
        pub_result = result[0]
        # Access counts from the classical register 'c'
        counts_bitstrings = pub_result.data.c.get_counts()
        return counts_bitstrings
    except ImportError:
        pass
    except Exception as e:
        print(f"StatevectorSampler error: {e}")

    # 3. Fallback: Random (if everything fails)
    print("Warning: Using simulated random counts (no real simulator available)")
    n_qubits = qc.num_qubits
    counts = {}
    for _ in range(shots):
        bitstring = ''.join(str(np.random.randint(0, 2)) for _ in range(n_qubits))
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts

def get_expectation(graph, p=1):
    """
    Returns a function that takes parameters theta and returns the expectation value.
    """
    def execute_circ(theta):
        qc = create_qaoa_circuit(graph, theta)
        
        # Execute the circuit
        counts = simulate_circuit(qc, shots=1024)
        
        # Qiskit may return bitstrings in reverse order; try both orderings
        # and use whichever gives better structure
        counts_corrected = {}
        for bitstring, count in counts.items():
            # Try to reverse if it looks reversed (optional)
            if len(bitstring) == len(graph.nodes):
                bitstring_reversed = bitstring[::-1]
                counts_corrected[bitstring_reversed] = count
            else:
                counts_corrected[bitstring] = count
        
        return compute_expectation(counts_corrected, graph)
    
    return execute_circ

def main():
    # 1. Define the Graph
    # A simple 4-node graph (0-1, 1-2, 2-3, 3-0) - A square
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    
    print("="*70)
    print("QAOA MaxCut Problem Solver")
    print("="*70)
    print(f"\nGraph Nodes: {list(G.nodes())}")
    print(f"Graph Edges: {list(G.edges())}")
    print(f"Maximum Possible Cut Size: {len(G.edges())}")
    
    # 2. Optimize
    p = 1  # Depth of the circuit
    print(f"\nStarting QAOA optimization with p={p}...")
    print("This may take a minute or two...\n")
    expectation = get_expectation(G, p=p)
    
    # Initial guess for parameters (beta, gamma)
    # 2 parameters for p=1
    init_params = [1.0, 1.0]
    
    res = minimize(expectation, init_params, method='COBYLA', 
                   options={'maxiter': 50, 'tol': 1e-4})
    
    print("\n" + "="*70)
    print("Optimization Result:")
    print("="*70)
    print(f"Success: {res.success}")
    print(f"Best Expectation Value: {res.fun:.6f}")
    print(f"Optimal Parameters: beta={res.x[0]:.6f}, gamma={res.x[1]:.6f}")
    print(f"Function Evaluations: {res.nfev}")
    
    # 3. Run final circuit with optimal parameters
    print("\n" + "="*70)
    print("Final Results with Optimal Parameters (5000 shots):")
    print("="*70)
    
    optimal_theta = res.x
    qc = create_qaoa_circuit(G, optimal_theta)
    
    counts = simulate_circuit(qc, shots=5000)
    
    # Correct bitstring order
    counts_corrected = {}
    for bitstring, count in counts.items():
        if len(bitstring) == len(G.nodes):
            bitstring_reversed = bitstring[::-1]
            counts_corrected[bitstring_reversed] = count
        else:
            counts_corrected[bitstring] = count
    
    # Sort by count and display top states
    sorted_counts = sorted(counts_corrected.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'State':<10} {'Count':<8} {'Probability':<12} {'Cut Size':<10}")
    print("-" * 45)
    
    for bitstring, count in sorted_counts[:10]:
        prob = count / 5000
        cut_size = -maxcut_obj(bitstring, G)
        print(f"|{bitstring}> {count:<7} {prob:<12.4f} {cut_size:<10}")
    
    # Calculate overall expectation and best cut found
    exp_value = compute_expectation(counts_corrected, G)
    best_state = sorted_counts[0][0]
    best_cut = -maxcut_obj(best_state, G)
    
    print(f"\nExpectation Value: {exp_value:.6f}")
    print(f"Best State Found: |{best_state}> with Cut Size: {best_cut}")
    print(f"Maximum Possible Cut Size: {len(G.edges())}")
    print(f"Approximation Ratio: {best_cut}/{len(G.edges())} = {100*best_cut/len(G.edges()):.1f}%")
    print("="*70)

if __name__ == "__main__":
    main()