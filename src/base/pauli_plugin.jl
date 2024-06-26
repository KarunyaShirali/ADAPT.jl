import ..ADAPT

import .MyPauliOperators
import .MyPauliOperators: SparseKetBasis
import .MyPauliOperators: FixedPhasePauli, ScaledPauli, Pauli
import .MyPauliOperators: ScaledPauliVector, PauliSum
# TODO: Replace `MyPauliOperators` with `PauliOperators` throughout, once merged.

import LinearAlgebra: mul!, dot

import KrylovKit

# TODO: Usually when we take `AbstractAnsatz` we mean `AbstractAnsatz{F,G<:AnyPauli}`.

AnyPauli = Union{Pauli, ScaledPauli, PauliSum, ScaledPauliVector}
#= NOTE: ANY Pauli is a little strong.
We don't actually support FixedPhasePauli in this library.

(This is just because it's rather confusingly defined,
    so I want users to be explicit and just use Pauli.
    I bet that's what Nick intended also.)
=#

ADAPT.typeof_energy(::AnyPauli) = Float64
#= NOTE:
    This type assertion assumes two things:
    1. Scaled AbstractPaulis are hard-coded to used `ComplexF64`.
    2. The observable is Hermitian (so its energy is guaranteed to be real).

    Strictly speaking, using the plain `Pauli` type undermines both assumptions,
        since half of them are anti-Hermitian and they don't have *any* float type.
    So...try to avoid using `Pauli` as an `Observable`?
    It should work fine if the phase is real, but...it'd be less than robust...
=#

function ADAPT.evolve_state!(
    G::Pauli,
    θ::ADAPT.Parameter,
    Ψ::ADAPT.QuantumState,
)
    angle = -θ * MyPauliOperators.get_phase(G) * MyPauliOperators.get_phase(G.pauli)'
    MyPauliOperators.cis!(Ψ, G.pauli, angle)
    Ψ isa SparseKetBasis && MyPauliOperators.clip!(Ψ)
    return Ψ
end

function ADAPT.evolve_state!(
    G::ScaledPauli,
    θ::ADAPT.Parameter,
    Ψ::ADAPT.QuantumState,
)
    angle = -θ * G.coeff * MyPauliOperators.get_phase(G.pauli)'
    MyPauliOperators.cis!(Ψ, G.pauli, angle)
    Ψ isa SparseKetBasis && MyPauliOperators.clip!(Ψ)
    return Ψ
end

function ADAPT.evolve_state!(
    G::ScaledPauliVector,
    θ::ADAPT.Parameter,
    Ψ::ADAPT.QuantumState,
)
    for P in G
        ADAPT.evolve_state!(P, θ, Ψ)
    end
    return Ψ
end

function ADAPT.evolve_state!(
    G::PauliSum,
    θ::ADAPT.Parameter,
    Ψ::AbstractVector,
)
    Ψ .= KrylovKit.exponentiate(x -> G * x, -im*θ, Ψ; ishermitian=true)[1]
    return Ψ
end


# TEMP: Working function, with type instability.
function ADAPT.evaluate(
    H::AnyPauli,
    Ψ::ADAPT.QuantumState,
)
    return real(MyPauliOperators.expectation_value(H, Ψ))
end


"""
    partial(
        index::Int,
        ansatz::AbstractAnsatz,
        observable::Observable,
        reference::QuantumState,
    )

The partial derivative of a cost-function with respect to the i-th parameter in an ansatz.

The ansatz is assumed to apply a unitary rotation `exp(-iθG)`,
    where `G` is the (Hermitian) generator,
    and generators with a lower index are applied to the state earlier.
Ansatz sub-types may change both behaviors.

# Parameters
- `index`: the index of the parameter to calculate within `ansatz`
- `ansatz`: the ADAPT state
- `H`: the object defining the cost-function
- `ψ0`: an initial quantum state which the `ansatz` operates on

# Returns
- a number of type `typeof_energy(observable)`.

"""
function ADAPT.partial(
    index::Int,
    ansatz::ADAPT.AbstractAnsatz,
    observable::AnyPauli,
    reference::ADAPT.QuantumState,
)
    state = deepcopy(reference)

    # PARTIAL EVOLUTION
    for i in 1:index-1
        generator, parameter = ansatz[i]
        ADAPT.evolve_state!(generator, parameter, state)
    end

    # REFLECTION
    generator, parameter = ansatz[index]
    costate = __make__costate(generator, parameter, state)
    ADAPT.evolve_state!(generator, parameter, state)

    # FINISH EVOLUTION
    for i in 1+index:length(ansatz)
        generator, parameter = ansatz[i]
        ADAPT.evolve_state!(generator, parameter, state)
        ADAPT.evolve_state!(generator, parameter, costate)
    end

    return 2 * real(MyPauliOperators.braket(observable, costate, state))
end


"""
    __make__costate(G, x, Ψ)

Compute ∂/∂x exp(ixG) |ψ⟩.

"""
function __make__costate(G, x, Ψ)
    costate = -im * G * Ψ
    ADAPT.evolve_state!(G, x, costate)
    return costate
end

"""
    __make__costate(G::ScaledPauliVector, x, Ψ)

Compute ∂/∂x exp(ixG) |ψ⟩.

Default implementation just applies -iG to Ψ then evolves.
That's fine as long as the evolution is exact.
But evolution is not exact if `G` is a `ScaledPauliVector` containing non-commuting terms.
In such a case, the co-state must be more complicated.

"""
function __make__costate(G::ScaledPauliVector, x, Ψ::SparseKetBasis)
    costate = zero(Ψ)
    for (i, P) in enumerate(G)
        term = deepcopy(Ψ)
        for j in 1:i-1; ADAPT.evolve_state!(G[j], x, term); end         # RIGHT EVOLUTION
        term = -im * P * term                                           # REFLECTION
        #= TODO: Hypothetically could implement mul! for SparseKetBasis someday? =#
        for j in i:length(G); ADAPT.evolve_state!(G[j], x, term); end   # LEFT EVOLUTION
        sum!(costate, term)
    end
    MyPauliOperators.clip!(costate)
    return costate
end

function __make__costate(G::ScaledPauliVector, x, Ψ::AbstractVector)
    costate = zero(Ψ)
    work_l = Array{eltype(Ψ)}(undef, size(Ψ))
    work_r = Array{eltype(Ψ)}(undef, size(Ψ))
    for (i, P) in enumerate(G)
        work_r .= Ψ
        for j in 1:i-1; ADAPT.evolve_state!(G[j], x, work_r); end       # RIGHT EVOLUTION
        mul!(work_l, P, work_r); work_l .*= -im                         # REFLECTION
        for j in i:length(G); ADAPT.evolve_state!(G[j], x, work_l); end # LEFT EVOLUTION
        costate .+= work_l
    end
    return costate
end

function ADAPT.gradient!(
    result::AbstractVector,
    ansatz::ADAPT.AbstractAnsatz,
    observable::AnyPauli,
    reference::ADAPT.QuantumState,
)
    ψ = ADAPT.evolve_state(ansatz, reference)   # FULLY EVOLVED ANSATZ |ψ⟩
    λ = observable * ψ                          # CALCULATE |λ⟩ = H |ψ⟩

    for i in reverse(eachindex(ansatz))
        G, θ = ansatz[i]
        ADAPT.evolve_state!(G', -θ, ψ)          # UNEVOLVE BRA
        σ = __make__costate(G, θ, ψ)            # CALCULATE |σ⟩ = exp(-iθG) (-iG) |ψ⟩
        result[i] = 2 * real(dot(σ, λ))         # CALCULATE GRADIENT ⟨λ|σ⟩ + h.t.
        ADAPT.evolve_state!(G', -θ, λ)          # UNEVOLVE KET
    end

    return result
end

##########################################################################################
#= Improved scoring for vanilla ADAPT. =#

function ADAPT.calculate_score(
    ansatz::ADAPT.AbstractAnsatz,
    ::ADAPT.Basics.VanillaADAPT,
    generator::AnyPauli,
    observable::AnyPauli,
    reference::ADAPT.QuantumState,
)
    state = ADAPT.evolve_state(ansatz, reference)
    return abs(MyPauliOperators.measure_commutator(generator, observable, state))
end

function ADAPT.calculate_scores(
    ansatz::ADAPT.AbstractAnsatz,
    adapt::ADAPT.Basics.VanillaADAPT,
    pool::AnyPauli,
    observable::AnyPauli,
    reference::ADAPT.QuantumState,
)
    state = ADAPT.evolve_state(ansatz, reference)
    scores = Vector{ADAPT.typeof_score(adapt)}(undef, length(pool))
    for i in eachindex(pool)
        scores[i] = abs(MyPauliOperators.measure_commutator(pool[i], observable, state))
    end
    return scores
end