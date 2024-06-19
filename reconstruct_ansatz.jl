import ADAPT
import CSV
import DataFrames
import DataFrames: groupby
import Serialization
import LinearAlgebra: norm

n = 6

# READ IN HAMILTONIAN
serialized_H = "qaoa_dataset/Hamiltonian1_n_"*string(n)*"_Hamiltonian"
H = Serialization.deserialize(serialized_H)

# READ IN ADAPT-QAOA RESULTS
results_file = "qaoa_dataset/Hamiltonian1_n_"*string(n)*"_adaptqaoa_results.csv"
csv = CSV.File(results_file); my_df = DataFrames.DataFrame(csv)
gd = groupby(my_df, :run)

# INITIALIZE THE ANSATZ 
run = 1 # the index of the run for which you want to reconstruct the ansatz
ψ0 = ones(ComplexF64, 2^n) / sqrt(2^n); ψ0 /= norm(ψ0) # initialize ψ0
ansatz = ADAPT.ADAPT_QAOA.QAOAAnsatz(0.1, H) # initialize ansatz

# BUILD OUT THE OPERATOR POOL
pooltype = gd[run][1,:pooltype]; 
if pooltype=="qaoa_double_pool" 
    pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n)
end

# RECONSTRUCT ANSATZ
for row in eachrow(gd[run])
    push!(ansatz, pool[row.:generator_index_in_pool] => 0.0)
    # NOTE: this step adds both H and the pool operator to the ansatz
end
angles = collect(Iterators.flatten(zip(gd[run][!,:γ_coeff], gd[run][!,:β_coeff])))
ADAPT.bind!(ansatz, angles)  #= <- this is your reconstructed ansatz =#

# TEST: EVALUATE FINAL ENERGY - SHOULD MATCH LAST ENERGY FOR THAT 'run'
ψEND = ADAPT.evolve_state(ansatz, ψ0)
E_final = ADAPT.evaluate(H, ψEND)
println("final energy = $E_final")