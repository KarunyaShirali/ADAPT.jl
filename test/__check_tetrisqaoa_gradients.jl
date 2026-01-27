#= Run TETRIS-ADAPT with a TetrisQAOAAnsatz. =#
import ADAPT
import Graphs
import PauliOperators
import PauliOperators: ScaledPauliVector, FixedPhasePauli, KetBitString, SparseKetBasis, Pauli
import LinearAlgebra: norm
import Test: @test
import Random; Random.seed!(0)

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

# BUILD OUT THE POOL
pool = ADAPT.ADAPT_QAOA.QAOApools.qaoa_double_pool(n)

# CONSTRUCT A REFERENCE STATE
ψ0 = ones(ComplexF64, 2^n) / sqrt(2^n); ψ0 /= norm(ψ0)

# INITIALIZE THE ANSATZ AND TRACE
gamma0 = 0.1; println("gamma0 = $(gamma0)")
ansatz = ADAPT.ADAPT_QAOA.TetrisQAOAAnsatz(gamma0, pool, H)
trace = ADAPT.Trace()

# EVOLVE BY A BIT
evolver = pool[3]
# display(ψ0); display(evolver)
ADAPT.evolve_state!(evolver, -1.613, ψ0)
# display(ψ0); println("\n"^4)

# ADD ARBITRARY DISJOINT OPERATORS TO THE ANSATZ
push!(ansatz.γ_values, ansatz.γ0)
push!(ansatz.γ_layers, 1+length(ansatz.parameters))
push!(ansatz, pool[9] => -1.613) # operator for n = 6: yyIIII
push!(ansatz, pool[44] =>  0.237) # operator for n = 6: IIXXII
push!(ansatz, pool[67] => -0.373) # operator for n = 6: IIIIZy

push!(ansatz.γ_values, ansatz.γ0)
push!(ansatz.γ_layers, 1+length(ansatz.parameters))
push!(ansatz, pool[15] => 0.933) # operator for n = 6: ZIyIII
push!(ansatz, pool[2] =>  -2.194) # operator for n = 6: IXIIII
push!(ansatz, pool[65] => -0.356) # operator for n = 6: IIIIyy
# push!(ansatz, pool[3][1] => π/2)

# MEASURE GRADIENT ANALYTICALLY
g0 = zero(ADAPT.angles(ansatz))
@time ADAPT.gradient!(g0, ansatz, H, ψ0)
# gO = zero(g0)
# fillgO = i -> gO[i] = ADAPT.partial(i, ansatz, H, ψ0)
# @time foreach(fillgO, eachindex(ansatz))

# MEASURE GRADIENT NUMERICALLY
import FiniteDifferences
cfd = FiniteDifferences.central_fdm(5, 1)
costfn = ADAPT.make_costfunction(ansatz, H, ψ0)
gx = FiniteDifferences.grad(cfd, costfn, ADAPT.angles(ansatz))[1]

tolerance = 1e-10
@test norm(g0 .- gx) ≤ tolerance