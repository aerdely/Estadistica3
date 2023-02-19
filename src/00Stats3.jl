# Statistics III cumulative collection of functions for the course

# edf 
"""
    Fn(x::Real, xobs::Vector{<:Real})

Empirical distribution function evaluated at value `x` using 
the observed random sample given in vector `xobs`.

## Example
```
Fn(0.0, randn(1_000))
```
"""
function Fn(x::Real, xobs::Vector{<:Real})
    return count(xobs .≤ x) / length(xobs)
end

"""
    Fn(x::Vector{<:Real}, xobs::Vector{<:Real})

Empirical distribution function evaluated at each point of vector `x` 
using the observed random sample given in vector `xobs`.

## Example
```
Fn([-1.96, 0.0, 1.96], randn(10_000))
```
"""
function Fn(x::Vector{<:Real}, xobs::Vector{<:Real})
    m = length(x)
    n = length(xobs)
    v = zeros(m)
    for k ∈ 1:m
        v[k] = count(xobs .≤ x[k]) / n
    end
    return v
end

# Tn
"""
    Tn(interval::String, xobs::Vector{<:Real})

Nonparametric point estimation for the probability of an `interval` 
using the observed random sample in vector `xobs`.
## Example
```
Tn("]-1.96,1.96]", randn(1_000_000)) # ≈ 0.95
```
"""
function Tn(interval::String, xobs::Vector{<:Real}) 
    m = length(interval)
    brackets = ['[', ']']
    if !issubset([interval[1], interval[m]], brackets)
        error("Error: interval must start and end with brackets.")
        return nothing
    end
    icomma = 0
    for i ∈ 1:m
        if interval[i] == ','
            icomma = i
        end
    end
    if icomma == 0
        error("Error: interval |a,b| extremes must be separated by a comma.")
        return nothing
    end
    a = parse(Float64, interval[2:(icomma - 1)])
    b = parse(Float64, interval[(icomma + 1):(m-1)])
    if a > b
        error("Error: interval |a,b| extremes must satisfy a ≤ b.")
        return nothing
    end
    n = length(xobs)
    if interval[1] == ']' && interval[m] == ']'
        tn = count(a .< xobs .≤ b) / n
    end
    if interval[1] == ']' && interval[m] == '['
        tn = count(a .< xobs .< b) / n
    end
    if interval[1] == '[' && interval[m] == ']'
        tn = count(a .≤ xobs .≤ b) / n
    end
    if interval[1] == '[' && interval[m] == '['
        tn = count(a .≤ xobs .< b) / n
    end
    return tn
end

# cuantil_puntual
"""
    cuantil_puntual(α::Real, obs::Vector{<:Real})

Empirical `α` quantile point estimation using the observed random sample in a vector `obs`.

# Example
```
cuantil_puntual(0.97725, randn(100000)) # theoretically it's 2.0 approx
```
"""
function cuantil(α::Real, xobs::Vector{<:Real})
    obs = sort(xobs)
    n = length(obs)
    obsmin = minimum(obs)
    obsmax = maximum(obs)
    orden = (n+1)*α
    if orden < 1
        cuantil = obsmin
        println("Aviso: cuantil fuera de rango muestral, es menor.")
        return cuantil
    end
    if orden > n
        cuantil = obsmax
        println("Aviso: cuantil fuera de rango muestral, es mayor.")
        return cuantil
    end
    if orden == round(orden)
        j = Int(orden)
        cuantil = obs[j]
        return cuantil
    end
    # interpolar:
    j = Int(floor(orden))
    cuantil = (j + 1  - orden)*obs[j] + (orden - j)*obs[j+1]
    return cuantil
end

### Table of contents

function Stats3()
    println("Stats3.jl")
    println("=========")
    println("Main functions: Fn  Tn  cuantil")
    println("Table of contents: Stats3()")
end
;