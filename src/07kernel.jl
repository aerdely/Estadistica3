### Estimación de funciones de densidad por kernel de suavizamiento

using Distributions, Plots

begin
    α = 0.3
    μ1 = -4; σ1 = 1
    μ2 = 3; σ2 = 2
    X = Normal(μ1, σ1)
    Y = Normal(μ2, σ2)
    B = Bernoulli(α)
end;

begin
    n = 10_000
    rB = rand(B, n)
    rX = rand(X, n)
    rY = rand(Y, n)
    rZ = rB .* rX .+ (1 .- rB) .* rY
end;

begin
    histogram(rZ, normalize = true, legend = false, color = "yellow")
    xaxis!("Z"); yaxis!("densidad")
end

function kernel(h, K, xobs)
    n = length(xobs)
    g(x) = sum(K.((x .- xobs)/h)) / (n*h)
end

begin
    K(x) = pdf(Normal(0, 1), x)
    f = kernel(0.5, K, rZ)
    xx = range(minimum(rZ), maximum(rZ), length = 1_000)
    plot!(xx, f.(xx), lw = 2)
end
