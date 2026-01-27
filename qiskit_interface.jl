module QiskitInterface
    import ADAPT
    using PyCall

    """
        __init__()

    Function to initialize the Python interface with qiskit. Defines a (Python) 
    function to use qiskit to generate the Pauli exponential operators and transpile 
    to a desired gate set.
    """
    function __init__()
        py"""
        import numpy as np
        import qiskit
        from qiskit import transpile
        from qiskit import QuantumCircuit
        from qiskit.providers.fake_provider import GenericBackendV2
        from qiskit.qasm2 import dumps
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.quantum_info import Statevector
        from qiskit_ibm_runtime import Estimator

        def transpile_to_backend(n, ansatz_ops, ansatz_coeffs):

            backend = GenericBackendV2(n)
            circuit = QuantumCircuit(n)

            for op_index, op in enumerate(ansatz_ops, start=0):
                operator = SparsePauliOp(op, ansatz_coeffs[op_index])
                # build the evolution gate
                evo = PauliEvolutionGate(operator, time=-1.0)
                # plug it into a circuit
                circuit.append(evo, range(n))
            # print(circuit.draw())

            qc_basis = transpile(circuit, 
            backend, 
            # basis_gates=['cx', 'id', 'rz', 'sx', 'x', 'reset', 'delay', 'measure'],
            optimization_level = 0)
            print(qc_basis)
            CNOT_depth = qc_basis.depth(lambda instr: len(instr.qubits) > 1)

            # CNOT_count = qc_basis.count(lambda instr: len(instr.qubits) > 1)
            # qasm = dumps(qc_basis)

            return CNOT_depth #, CNOT_count

        def validate_energy(Hamiltonian_Paulis, Hamiltonian_coefficients, ansatz_ops, ansatz_coeffs, reference, n):
            backend = GenericBackendV2(n) # set up a backend
            circuit = QuantumCircuit(n) # initialize the quantum circuit
            # note that it starts in the |000...0> state

            ψ0 = Statevector(reference)
            circuit.initialize(ψ0)
            # circuit.x(1)
            # circuit.x(2)
            # for qubit in range(1,n+1):
                # circuit.h(qubit)

            estimator = Estimator(backend, options={"default_shots": int(1e4)}) # define an estimator and shot number

            H = list(zip(Hamiltonian_Paulis, Hamiltonian_coefficients)) # Hamiltonian as list of Pauli strings, coefficients
            H_op = SparsePauliOp.from_list(H) # Hamiltonian as sparse Pauli operator

            for op_index, op in enumerate(ansatz_ops, start=0):
                operator = SparsePauliOp(op, ansatz_coeffs[op_index])
                evo = PauliEvolutionGate(operator, time=-1.0) # build the evolution gate
                ''' Note: the negative sign here in the time parameter 
                is due to a difference in sign conventions between qiskit and ADAPT.jl.
                In ADAPT.jl, we write the unitary evolution as exp(-iθG), whereas in qiskit it is exp(+iθG).'''
                circuit.append(evo, range(n)) # plug it into a circuit
            # print(circuit.draw())

            qc_basis = transpile(circuit, 
                                backend, 
                                optimization_level = 0)

            job = estimator.run([(qc_basis, H_op)])
            
            measured_energy = job.result()[0].data.evs

            return measured_energy
        """
    end

    import ADAPT
    import PauliOperators: ScaledPauliVector, FixedPhasePauli, PauliSum, get_phase
    
    function transpile_CNOTS(ansatz, num_qubits)
        ops = Vector{Vector{String}}()
        coeffs = Vector{Vector{Float64}}()
        for (op, coeff) in ansatz
            op_strs, op_coeffs = pauliOp_to_str(op)
            op_coeffs *= coeff
            push!(coeffs, op_coeffs)
            push!(ops, op_strs)
        end

        cnot_depth = py"transpile_to_backend"(num_qubits, ops, coeffs)

        return cnot_depth
    end

    function validate_energy(Hamiltonian::PauliSum, ansatz, reference::Vector{ComplexF64}, num_qubits)
        ops = Vector{Vector{String}}()
        coeffs = Vector{Vector{Float64}}()
        for (op, coeff) in ansatz
            op_strs, op_coeffs = pauliOp_to_str(op)
            op_coeffs *= coeff
            push!(coeffs, op_coeffs)
            push!(ops, op_strs)
        end

        H_ops = Vector{String}()
        H_coeffs = Vector{Float64}()
        for (H_op, H_op_coeff) in Hamiltonian.ops
            H_op_str, H_op_phase = pauliOp_to_str(H_op)
            H_op_coeff *= H_op_phase

            push!(H_coeffs, real(H_op_coeff))
            push!(H_ops, H_op_str)
        end

        energy = py"validate_energy"(H_ops, H_coeffs, ops, coeffs, reference, num_qubits)

        return energy
    end

    function validate_energy(Hamiltonian::ScaledPauliVector, ansatz, reference::Vector{ComplexF64}, num_qubits)
        ops = Vector{Vector{String}}()
        coeffs = Vector{Vector{Float64}}()
        for (op, coeff) in ansatz
            op_strs, op_coeffs = pauliOp_to_str(op)
            op_coeffs *= coeff
            push!(coeffs, op_coeffs)
            push!(ops, op_strs)
        end

        H_ops = Vector{String}()
        H_coeffs = Vector{Float64}()
        H_ops, H_coeffs = pauliOp_to_str(Hamiltonian)

        energy = py"validate_energy"(H_ops, H_coeffs, ops, coeffs, reference, num_qubits)

        return energy
    end

    function pauliOp_to_str(spv::ScaledPauliVector)
        ops = Vector{String}()
        coeffs = Vector{Float64}()
        strs = [string(sp.pauli) for sp in spv]
        coeffs = [(get_phase(sp.pauli))*sp.coeff for sp in spv]
        if !iszero(imag.(coeffs))
            println("ERROR: Coefficient is imaginary after correcting for phase.")
            exit()
        end
        ops = replace.(strs, "y" => "Y")
    
        return ops, coeffs
    end

    function pauliOp_to_str(fpp::FixedPhasePauli)
        str = string(fpp)
        coeff = get_phase(fpp)
        if !iszero(imag(coeff))
            println("ERROR: Coefficient is imaginary after correcting for phase.")
            exit()
        end
        op = replace(str, "y" => "Y")
    
        return op, coeff
    end

    function pauliOp_to_str(qaoa_obs::ADAPT.ADAPT_QAOA.QAOAObservable)
        ops = Vector{String}()
        coeffs = Vector{Float64}()
        strs = [string(sp.pauli) for sp in qaoa_obs.spv]
        coeffs = [(get_phase(sp.pauli))*sp.coeff for sp in qaoa_obs.spv]
        if !iszero(imag.(coeffs))
            println("ERROR: Coefficient is imaginary after correcting for phase.")
            exit()
        end
        ops = replace.(strs, "y" => "Y")
        
        return ops, coeffs
    end

    function revert_endianness(ψ::Vector{ComplexF64})
        n = Int(log2(length(ψ)))
        ψ_revert_endianness = zeros(ComplexF64, size(ψ))
        for i in 0:(length(ψ)-1)
            bitstring = join(digits(i,base=2, pad=n))
            ψ_revert_endianness[parse(Int128, bitstring, base=2)+1]=ψ[i+1]
        end
        return ψ_revert_endianness
    end
end