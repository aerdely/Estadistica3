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

# EDA
"""
    EDA(fobj, valmin, valmax; iEnteros = zeros(Int, 0), tamgen = 1000, propselec = 0.3, difmax = 0.00001, maxiter = 1000)

`fobj` A real function of several variables to be minimized, where its argument is a vector or 1-D array.

`valmin, valmax` vectors or 1-D arrays of minimum and maximum values for the first generation.

`iEnteros` index of variables that must take integer values.

`tamgen` size of each generation.

`propselec` proportion of population to be selected.

`difmax` error tolerance.

`maxiter` maximum number of iterations.

# Example 1
```
f(x) = (x[1] - 5)^4 - 16(x[1] - 5)^2 + 5(x[1] - 5) + 120
EDA(f, [0], [9])
```

# Example 2
This non-negative function is clearly minimized at (5,-2).
```
f(z) = abs(z[1] - 5) + abs(z[2] + 2)
EDA(f, [-10, -10], [10, 10])
```

# Example 3
The same function but only allowing integer values:
```
EDA(f, [-10, -10], [10, 10], iEnteros = [1, 2])
```
"""
function EDA(fobj, valmin, valmax; iEnteros = zeros(Int, 0), tamgen = 1000,
             propselec = 0.3, difmax = 0.00001, maxiter = 1000)
    numiter = 1
    println("Iterando... ")
    numvar = length(valmin)
    nselec = Int(round(tamgen * propselec))
    G = zeros(tamgen, numvar)
    Gselec = zeros(nselec, numvar)
    for j ∈ 1:numvar
        G[:, j] = valmin[j] .+ (valmax[j] - valmin[j]) .* rand(tamgen)
    end
    if length(iEnteros) > 0
        for j ∈ iEnteros
            G[:, j] = round.(G[:, j])
        end
    end
    d(x, y) = sqrt(sum((x .- y) .^ 2))
    rnorm(n, μ, σ) = μ .+ (σ .* randn(n))
    promedio(x) = sum(x) / length(x)
    desvest(x) = sqrt(sum((x .- promedio(x)) .^ 2) / (length(x) - 1))
    fG = zeros(tamgen)
    maxGselec = zeros(tamgen)
    minGselec = zeros(tamgen)
    media = zeros(numvar)
    desv = zeros(numvar)
    while numiter < maxiter
        # evaluando función objetivo en generación actual:
        print(numiter, "\r")
        for i ∈ 1:tamgen
            fG[i] = fobj(G[i, :])
        end
        # seleccionando de generación actual:
        umbral = sort(fG)[nselec]
        iSelec = findall(fG .≤ umbral)
        Gselec = G[iSelec, :]
        for j ∈ 1:numvar
            maxGselec[j] = maximum(Gselec[:, j])
            minGselec[j] = minimum(Gselec[:, j])
            media[j] = promedio(Gselec[:, j])
            desv[j] = desvest(Gselec[:, j])
        end
        # salir del ciclo si se cumple criterio de paro:
        if d(minGselec, maxGselec) < difmax 
            break
        end
        # y si no se cumple criterio de paro, nueva generación:
        numiter += 1
        for j ∈ 1:numvar
            G[:, j] = rnorm(tamgen, media[j], desv[j])
        end
        if length(iEnteros) > 0
            for j ∈ iEnteros
                G[:, j] = round.(G[:, j])
            end
        end
    end
    println("...fin")
    fGselec = zeros(nselec)
    for i ∈ 1:length(fGselec)
        fGselec[i] = fobj(Gselec[i, :])
    end
    xopt = Gselec[findmin(fGselec)[2], :]
    if length(iEnteros) > 0
        for j ∈ iEnteros
            xopt[j] = round(xopt[j])
        end
    end
    fxopt = fobj(xopt)
    r = (x = xopt, fx = fxopt, iter = numiter)
    if numiter == maxiter
        println("Aviso: se alcanzó el máximo número de iteraciones = ", maxiter)
    end
    return r
end

# Bn
using Distributions
"""
    Bn(interval::String, obs, γ = 0.95)

Nonparametric point and 100γ% probability interval estimation for the probability of an `interval` using the observed random sample in a vector `obs` and the bayesian paradigm (g = 0.95 default value if not specified). `interval` is specified a string, where the first and last characters must be an open or closed bracket, that is `]` or `[` and the left and right extremes must be numbers separated by a comma.

Dependencies:
- internal: `Tn` `EDA`
- external: `Distributions.jl`

# Example
```
W = Normal(-2, 3)
cdf(W, 3) - cdf(W, 0) # P(0 < W < 3) 
wobs = rand(W, 1_000);
Bn("]0,3[", wobs, 0.99)
```
"""
function Bn(interval::String, obs, γ = 0.95)
    # using: Distributions.jl
    # Dependencies: Tn, EDA
    n = length(obs)
    tn = Tn(interval, obs)
    α, β = 1 + n*tn, 1 + n*(1 - tn) # posterior parameters
    Θ = Beta(α, β) # posterior distribution
    θmedia, θmediana = mean(Θ), median(Θ)
    h(z) = (quantile(Θ, γ + cdf(Θ, z[1])) - z[1]) * Inf^(z[1] > quantile(Θ, 1 - γ))
    sol = EDA(h, [0], [quantile(Θ, 1- γ)])
    θ₁ = sol[1][1]
    θ₂ = quantile(Θ, γ + cdf(Θ, θ₁))
    estimación = (insesgado = tn, media = θmedia, mediana = θmediana, intervalo = (θ₁, θ₂))
    return estimación
end


### Table of contents

function Stats3()
    println("Stats3.jl")
    println("=========")
    println("Main functions: Fn  Tn  Bn")
    println("Auxiliary: EDA")
    println("Table of contents: Stats3()")
end
;