### Estimation of Distribution Algorithms (EDAs)

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


## Example 1
## Minimize f over [0,9]
begin
    f(x) = (x[1] - 5)^4 - 16(x[1] - 5)^2 + 5(x[1] - 5) + 120
    sol = EDA(f, [0], [9])
end

using Plots, LaTeXStrings

begin    
    x = collect(range(0, 9, length = 1_000))
    plot(x, f.(x), legend = false, lw = 3, xlabel = L"x", ylabel = L"f(x)")
    scatter!([sol[1]], [sol[2]])
end


## Example 2
## f(x,y) = |x - 5| + |y + 2| is clearly minimzed ay (x,y) = (5,-2) since f(x,y) ≥ 0
begin
    f(z) = abs(z[1] - 5) + abs(z[2] + 2)
    sol = EDA(f, [-10, -10], [10, 10])
end

# just over integer values:
sol = EDA(f, [-10, -10], [10, 10], iEnteros = [1, 2])


## Example 3
## Maximum Likelihood Estimation

using Distributions

begin
    X = Normal(-2, 5)
    rX = rand(X, 1_000)
    dnorm(x, μ, σ) = pdf(Normal(μ, max(σ, 0.00001)), x)
    logV(p) = -sum(log.(dnorm.(rX, p[1], p[2])))
    EDA(logV, [-10, 0], [10, 10])
end
