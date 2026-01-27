#= Run QAOA on a MaxCut Hamiltonian. =#

import Graphs
import ADAPT
import PauliOperators: ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis
import LinearAlgebra: norm

import Random; Random.seed!(0)

# DEFINE A GRAPH
n = 6

# EXAMPLE OF ERDOS-RENYI GRAPH
prob = 0.5
g = Graphs.erdos_renyi(n, prob)

# EXTRACT MAXCUT FROM GRAPH
e_list = ADAPT.Hamiltonians.get_unweighted_maxcut(g)

# BUILD OUT THE PROBLEM HAMILTONIAN
H = ADAPT.Hamiltonians.maxcut_hamiltonian(n, e_list)

println("Observable data type: ",typeof(H))

# EXACT DIAGONALIZATION (for accuracy assessment)
#= NOTE: Comment this out to avoid building exponentially-sized matrices!
    (but then you'll have to edit or remove the FloorStopper callback. =#
module Exact
    import ..H
    using LinearAlgebra
    Hm = Matrix(H); E, U = eigen(Hm) # NOTE: Comment out after first run when debugging.
    ψ0 = U[:,1]
    E0 = real(E[1])
end
println("Exact ground-state energy: ",Exact.E0)

# BUILD OUT THE POOL
pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n)
standard_mixer = pool[n+1]

println("Generator data type: ", typeof(pool[1]))
println("Note: in the current ADAPT-QAOA implementation, the observable and generators must have the same type.")

# CONSTRUCT A REFERENCE STATE
ψ0 = ones(ComplexF64, 2^n) / sqrt(2^n); ψ0 /= norm(ψ0)

# INITIALIZE THE ANSATZ AND TRACE
ansatz = ADAPT.ADAPT_QAOA.QAOAAnsatz(0.1, H)
# the first argument (a hyperparameter) can in principle be set to values other than 0.1
trace = ADAPT.Trace()

# SELECT THE PROTOCOLS
vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)
    #= NOTE: Add `iterations=10` to set max iterations per optimization loop. =#

callbacks = [
    ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_generator, :selected_score, :scores), 
    ADAPT.Callbacks.Printer(:energy),
]

# CONSTRUCT THE ANSATZ
layers = n
for i=1:layers
    push!(ansatz, standard_mixer => 0.0)
end
ADAPT.set_optimized!(ansatz, false)

# RUN THE OPTIMIZATION ALGORITHM
result = ADAPT.optimize!(ansatz, trace, vqe, H, ψ0, callbacks)
ψEND = ADAPT.evolve_state(ansatz, ψ0)
println("VQE Converged? $(result.g_converged)")
E0 = ADAPT.evaluate(H, ψEND); println(E0)
rel_energy_err = abs((Exact.E0 - E0)/Exact.E0)