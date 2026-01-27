#= Run TETRIS-ADAPT with a TetrisQAOAAnsatz. =#

import Graphs
import ADAPT
import PauliOperators
import PauliOperators: ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis, Pauli
import LinearAlgebra: norm
import CSV
import DataFrames
import Serialization
import Random; Random.seed!(0)

"""
    ModalSampleTracer()

At each adaptation, identify the most likely bitstring and save it as an integer.

In the context of QAOA, this identifies the most reasonable partition.

"""
struct ModalSampleTracer <: ADAPT.AbstractCallback end

function (tracer::ModalSampleTracer)(
    ::ADAPT.Data, ansatz::ADAPT.AbstractAnsatz, trace::ADAPT.Trace,
    ::ADAPT.AdaptProtocol, ::ADAPT.GeneratorList,
    ::ADAPT.Observable, ψ0::ADAPT.QuantumState,
)
    ψ = ADAPT.evolve_state(ansatz, ψ0)      # THE FINAL STATEVECTOR
    imode = argmax(abs2.(ψ))                # MOST LIKELY INDEX
    zmode = imode-1                         # MOST LIKELY BITSTRING (as int)

    push!( get!(trace, :modalsample, Any[]), zmode )
    return false
end

# DEFINE A GRAPH
n = 6

# EXAMPLE OF ERDOS-RENYI GRAPH
prob = 0.5; g = Graphs.erdos_renyi(n, prob)

# EXTRACT MAXCUT FROM GRAPH
e_list = ADAPT.Hamiltonians.get_unweighted_maxcut(g)

# BUILD OUT THE PROBLEM HAMILTONIAN
H_spv = ADAPT.Hamiltonians.maxcut_hamiltonian(n, e_list)

# Wrap in a QAOAObservable view.
H = ADAPT.ADAPT_QAOA.QAOAObservable(H_spv)

# EXACT DIAGONALIZATION (for accuracy assessment)
#= NOTE: This block now scales as well as evolving the ansatz,
    so there is no need to comment it out. =#
module Exact
    import ..PauliOperators
    import ..H, ..n
    Emin = Ref(Inf); ketmin = Ref(PauliOperators.KetBitString{n}(0))
    for v in 0:1<<n-1
        ket = PauliOperators.KetBitString{n}(v)
        vec = PauliOperators.SparseKetBasis{n,ComplexF64}(ket => 1)
        Ev = real((H*vec)[ket])
        if Ev < Emin[]
            Emin[] = Ev
            ketmin[] = ket
        end
    end
    ψ0 = Vector(PauliOperators.SparseKetBasis{n,ComplexF64}(ketmin[] => 1))
    E0 = Emin[]

    ρ = abs2.(ψ0)                           # THE FINAL PROBABILITY DISTRIBUTION
    pmax, imax = findmax(ρ)
    ketmax = PauliOperators.KetBitString(n, imax-1) # THE MOST LIKELY BITSTRING
end
println("Exact ground-state energy: ",Exact.E0)
println("Best cut: ",Exact.ketmax)

# BUILD OUT THE POOL
pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n); poolstr = "qaoa_double_pool"

# CONSTRUCT A REFERENCE STATE
ψ0 = ones(ComplexF64, 2^n) / sqrt(2^n); ψ0 /= norm(ψ0)

# INITIALIZE THE ANSATZ AND TRACE
gamma0 = 0.1; println("gamma0 = $(gamma0)")
ansatz = ADAPT.ADAPT_QAOA.TetrisQAOAAnsatz(gamma0, pool, H)

# the first argument (a hyperparameter) can in principle be set to values other than 0.1
trace = ADAPT.Trace()

# SELECT THE PROTOCOLS
adapt_gradient_threshold = 1e-3
adapt = ADAPT.TETRIS_ADAPT.TETRISADAPT(adapt_gradient_threshold) 
# number argument in TETRISADAPT should be *preferably* equal to the ADAPT gradient threshold
vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)
    #= NOTE: Add `iterations=10` to set max iterations per optimization loop. =#

# SELECT THE CALLBACKS
callbacks = [
    ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score),
    ADAPT.Callbacks.ParameterTracer(),
    ModalSampleTracer(),
    ADAPT.Callbacks.Printer(:energy),
    ADAPT.Callbacks.ScoreStopper(adapt_gradient_threshold),
    ADAPT.Callbacks.ParameterStopper(100),
    ADAPT.Callbacks.FloorStopper(0.5, Exact.E0),
    ADAPT.Callbacks.SlowStopper(1.0, 3),
]

# RUN THE ALGORITHM
success = ADAPT.run!(ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)
println(success ? "Success!" : "Failure - optimization didn't converge.")

# RESULTS
results_df = DataFrames.DataFrame()

if success
    # SAVE THE TRACE
    indices = [ansatz.γ_layers[i]:(ansatz.γ_layers[i+1]-1) for i in eachindex(ansatz.γ_layers)[1:end-1]]
    push!(indices, ansatz.γ_layers[end]:length(ansatz.parameters))
    parameters = [ansatz.parameters[i] for i in indices]

    # :γ_coeff => the γ coefficients (for the observable)
    # :selected_index => selected mixer indices from the pool
    # :β_coeff => the β coefficients (for the mixers)
    # :energy => energy after each ADAPT step
    # :bitstring => Most likely bitstrings after each adaption

    df = DataFrames.DataFrame(#:pooltype => poolstr,
            :gamma_coeff => ansatz.γ_values,
            :selected_index => trace[:selected_index][1:end-1], 
            :beta_coeff => parameters,
            :energy => trace[:energy][trace[:adaptation][2:end]],
            :bitstring => [string(el[end-n+1:end]) for el in bitstring.(trace[:modalsample][1:end-1])])

    # WRITE THE ADAPT-QAOA RESULTS TO A FILE
    H_asdict = Dict(string(sp.pauli) => sp.coeff for sp in H.spv)
    Serialization.serialize("adaptqaoa_Hamiltonian_n_"*string(n)*"_gamma0_"*string(gamma0), H_asdict)

    results_file = "adaptqaoa_results_n_"*string(n)*"_gamma0_"*string(gamma0)*".csv"
    CSV.write(results_file, df) 
end