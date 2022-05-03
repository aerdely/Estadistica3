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

# cuantil_puntual
"""
    cuantil_puntual(α::Real, obs::Vector{<:Real})

Empirical `α` quantile point estimation using the observed random sample in a vector `obs`.

# Example
```
cuantil_puntual(0.97725, randn(100000)) # theoretically it's 2.0 approx
```
"""
function cuantil_puntual(α::Real, xobs::Vector{<:Real})
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

# GoF
"""
    GoF(xobs, Fo, simFo; prueba = "AD", numsims = 100_000)

Goodness of fit test for an observed random sample `xobs` (as a vector),
a proposed distribution function `Fo` along with a function `simFo` which
simulates samples from `Fo` with a given size. `prueba` is a string indicating
`AD` for Anderson-Darling (default), `CM` for Cramér-von Mises, and `KS` for
Kolmogorov-Smirnov. `numsims` is the number of simulations to aproximate the
probability distribution of the chosen test statistic (100,000 by default).

## Example
```
xobs = rand(100) # random sample from a continuous Uniform(0,1) distribution
b = 1.1
Fo(x) = (0 < x < b) * x / b + 1*(x ≥ b) # Uniform(0,b) distribution function
simFo(n) = b * rand(n) # simulates a size n random sample from a Uniform(0,b)
GoF(xobs, Fo, simFo; prueba = "AD")
```
"""
function GoF(xobs, Fo, simFo; prueba = "AD", numsims = 100_000, msg = true)
    n = length(xobs)
    u(x) = Fo.(sort(x))
    i1 = collect(1:n)
    i0 = collect(0:(n-1))
    KS(x) = max(maximum(i1/n .- u(x)), maximum(u(x) .- i0/n))
    CM(x) = 1/(12*n) + sum((u(x) .- (2 .* i1 .- 1) ./ (2*n)).^2)
    AD(x) = -n - (1/n)*sum((2 .* i1 .- 1).*(log.(u(x)) .+ log.(1 .- sort(u(x), rev = true))))
    T = AD
    autores = "Anderson - Darling"
    if prueba == "KS"
        T = KS
        autores = "Kolmogorov - Smirnov"
    end
    if prueba == "CM"
        T = CM
        autores = "Cramér - von Mises"
    end
    tsim = zeros(numsims)
    for j ∈ 1:numsims
        x = simFo(n)
        tsim[j] = T(x)
    end
    tobs = T(xobs)
    pvalue = sum(tsim .> tobs) / numsims
    if msg == true
        println("p-valor de la prueba " * autores)
    end
    return pvalue
end

# Poligonal
"""
    poligonal(x::Vector{<:Real}, xobs::Vector{<:Real}; mínimo = minimum(xobs), máximo = maximum(xobs))

Polygonal approximation of a continuous distribution function at a given vector of values `x` based
on an observed random sample given by vector `xobs`. The optional parameter `mínimo` may be set to a value
smaller than the sample minimum, and `máximo` to a greater value than the sample maximum, if required.

## Example
```
xobs = randn(10_000)
x = [-1.96, 0, 1.96]
poligonal(x, xobs)
```
"""
function poligonal(x::Vector{<:Real}, xobs::Vector{<:Real}; mínimo = minimum(xobs), máximo = maximum(xobs))
    n = length(xobs)
    xord = sort(xobs)
    xp = (xord[1:(n-1)] .+ xord[2:n]) ./ 2
    xp = vcat(mínimo, xp, máximo)
    function g(z, xp)
        a = 0.0
        if z > xp[n+1]
            a = 1.0
        elseif z > xp[1]
            for k ∈ 2:(n+1)
                a += (xp[k-1] < z ≤ xp[k]) * ((k-1) - (xp[k] - z)/(xp[k] - xp[k-1]))
            end
            a /= n
        end
        return a
    end
    m = length(x)
    polivalores = zeros(m)
    for i ∈ 1:m
        polivalores[i] = g(x[i], xp)
    end
    return polivalores 
end

# poligonal_cuantiles
"""
    poligonal_cuantiles(u::Vector{<:Real}, xobs::Vector{<:Real}; mínimo = minimum(xobs), máximo = maximum(xobs))

Polygonal approximation of the quantile function of a continuous distribution given vector of values `0<u<1` based
on an observed random sample given by vector `xobs`. The optional parameter `mínimo` may be set to a value
smaller than the sample minimum, and `máximo` to a greater value than the sample maximum, if required.

## Example
```
xobs = randn(10_000)
u = [0.025, 0.5, 0.975]
poligonal_cuantiles(u, xobs)
```
"""
function poligonal_cuantiles(u::Vector{<:Real}, xobs::Vector{<:Real}; mínimo = minimum(xobs), máximo = maximum(xobs))
    n = length(xobs)
    xord = sort(xobs)
    xp = (xord[1:(n-1)] .+ xord[2:n]) ./ 2
    xp = vcat(mínimo, xp, máximo)
    function g(z, xp)
        if 0 < z < 1
            k = findmax(n*z .≤ collect(1:n))[2]
            return xp[k+1] - (xp[k+1] - xp[k])*(k - n*z)
        else
            return NaN 
        end
    end
    m = length(u)
    cuantiles = zeros(m)
    for i ∈ 1:m
        cuantiles[i] = g(u[i], xp)
    end
    return cuantiles
end


### Table of contents

function Stats3()
    println("Stats3.jl")
    println("=========")
    println("Main functions: Fn  Tn  Bn  cuantil_puntual  GoF  poligonal  poligonal_cuantiles")
    println("Auxiliary: EDA")
    println("Table of contents: Stats3()")
end
;