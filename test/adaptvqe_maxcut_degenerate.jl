#= Run ADAPT-VQE on a MaxCut Hamiltonian. =#
#= Sample various paths to convergence, by choosing
randomly from operators with degenerate gradients. =#

import Graphs
import ADAPT
import PauliOperators: ScaledPauliVector, FixedPhasePauli
import LinearAlgebra: norm
import CSV
import DataFrames
import Serialization

# DEFINE A GRAPH
n = 6

# EXAMPLE OF ERDOS-RENYI GRAPH
prob = 0.5
g = Graphs.erdos_renyi(n, prob)

# EXTRACT MAXCUT FROM GRAPH
e_list = ADAPT.Hamiltonians.get_unweighted_maxcut(g)

# BUILD OUT THE PROBLEM HAMILTONIAN
H = ADAPT.Hamiltonians.maxcut_hamiltonian(n, e_list)

# ANOTHER WAY TO BUILD OUT THE PROBLEM HAMILTONIAN
# d = 3 # degree of regular graph
# H = ADAPT.Hamiltonians.MaxCut.random_regular_max_cut_hamiltonian(n, d)
println("Observable data type: ",typeof(H))

# BUILD OUT THE POOL
pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n); pooltype = "qaoa_double_pool"

# ANOTHER POOL OPTION
# pool = ADAPT.Pools.two_local_pool(n); pooltype = "two_local_pool"

println("Generator data type: ", typeof(pool[1]))
println("Note: in the current ADAPT-QAOA implementation, the observable and generators must have the same type.")

# SELECT THE PROTOCOLS
adapt = ADAPT.VANILLA
vqe = ADAPT.OptimOptimizer(:BFGS; g_tol=1e-6)

# SELECT THE CALLBACKS
callbacks = [
    ADAPT.Callbacks.Tracer(:energy, :selected_index, :selected_score, :scores),
    ADAPT.Callbacks.ParameterTracer(),
    # ADAPT.Callbacks.Printer(:energy, :selected_index, :selected_score),
    ADAPT.Callbacks.ScoreStopper(1e-3),
    ADAPT.Callbacks.ParameterStopper(100),
    # ADAPT.Callbacks.FloorStopper(0.5, Exact.E0),
    # ADAPT.Callbacks.SlowStopper(1.0, 3),
]

# RUN MANY ADAPT-QAOA TRIALS, CHOOSING RANDOMLY WHEN THE GRADIENTS ARE DEGENERATE
results_df = DataFrames.DataFrame()
trials = 1
for trial_num = 1:trials
    # INITIALIZE THE REFERENCE STATE
    ψ0 = ones(ComplexF64, 2^n) / sqrt(2^n); ψ0 /= norm(ψ0)

    # INITIALIZE THE ANSATZ AND TRACE
    ansatz = ADAPT.Ansatz(Float64, pool)
    trace = ADAPT.Trace()

    # RUN THE ALGORITHM
    success = ADAPT.run!(ansatz, trace, adapt, vqe, pool, H, ψ0, callbacks)
    println(success ? "Success!" : "Failure - optimization didn't converge.")
    println(trace[:energy][trace[:adaptation][end]])
    # RESULTS
    if !success
        continue
    end
    
    # SAVE THE TRACE IF ADAPT WAS SUCCESSFUL
    df = DataFrames.DataFrame(:run => trial_num,
                              :pooltype => pooltype,
                              :generator_index_in_pool => trace[:selected_index][1:end-1], 
                              :coeff => ansatz.parameters,
                              :energy => trace[:energy][trace[:adaptation][2:end]])
    append!(results_df, df)


    # WRITE ANSATZ AS (GENERATOR, COEFFICIENT) LIST TO TXT FILE FOR EASY READABILITY
    generator_strings = Vector{String}()
    for _op in ansatz.generators
        push!(generator_strings, string(_op[1].pauli))
    end

    filename = "vqe_dataset/Hamiltonian1_n_"*string(n)*"_run_"*string(trial_num)*"_adaptvqe_ansatz.txt"
    open(filename, "w") do io
        writedlm(io,[generator_strings ansatz.parameters])
    end
end

# assign some index to the Hamiltonian
H_number = 1

# WRITE HAMILTONIAN TO CSV FILE
ham_file = "vqe_dataset/Hamiltonian"*string(H_number)*"_n_"*string(n)*"_Hamiltonian.csv"
CSV.write(ham_file, H)
# SAVE HAMILTONIAN - serialize for enhanced portability
ham_file = "vqe_dataset/Hamiltonian"*string(H_number)*"_n_"*string(n)*"_Hamiltonian"
Serialization.serialize(ham_file, H)

# WRITE THE ADAPT-VQE RESULTS TO A FILE
results_file = "vqe_dataset/Hamiltonian"*string(H_number)*"_n_"*string(n)*"_adaptvqe_results.csv"
CSV.write(results_file, results_df)
