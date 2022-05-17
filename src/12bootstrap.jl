### Inferencia no paramétrica mediante remuestreo con reemplazo (bootstrap)

begin
    using Distributions, Statistics, BenchmarkTools
    include("00Stats3.jl")
end


function bootstrap(estadístico, xobs; nB = 10_000, prob = 0.95)
    n = length(xobs)
    T = estadístico
    Tobs = T(xobs)
    Tsim = zeros(nB)
    for j ∈ 1:nB
        muestra = rand(xobs, n)
        Tsim[j] = T(muestra)
    end
    estim_puntual = mean(Tsim)
    estim_intervalo = quantile(Tsim, [(1 - prob)/2, (1 + prob)/2])
    estim = (puntual = estim_puntual, intervalo = estim_intervalo, valor = Tobs)
    return estim
end

function poligest(estadístico, xobs; nB = 10_000, prob = 0.95)
    n = length(xobs)
    T = estadístico
    Tobs = T(xobs)
    Tsim = zeros(nB)
    for j ∈ 1:nB
        muestra = poligonal_cuantiles(rand(n), xobs)
        Tsim[j] = T(muestra)
    end
    estim_puntual = mean(Tsim)
    estim_intervalo = quantile(Tsim, [(1 - prob)/2, (1 + prob)/2])
    estim = (puntual = estim_puntual, intervalo = estim_intervalo, valor = Tobs)
    return estim
end



## Ejemplo: m.a X1,...,Xn ~ Normal(μ, σ) 
## Estadístico: media muestral Xbarra ~ Normal(μ, σ / √n)

begin
    n = 100
    p = 0.95
    μ = 3; σ = 2
    X = Normal(μ, σ)
    Xbarra = Normal(μ, σ / √n)
    Xbarra_intervalo = quantile(Xbarra, [(1-p)/2, (1+p)/2])
end

xobs = rand(X, n)

estim_bootstrap = bootstrap(mean, xobs)

Xbarra_intervalo

estim_poligest = poligest(mean, xobs)

@btime estim_bootstrap = bootstrap(mean, xobs)
@btime estim_poligest = poligest(mean, xobs)
