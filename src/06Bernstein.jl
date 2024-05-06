### Aproximación de una función de cuantiles por polinomios de Bernstein

begin
    using Distributions, Plots, BenchmarkTools
    include("00Stats3.jl")
end


## Ejemplo: m.a ~ Gamma(2, 3)

# Simular muestra y calcular distribución teórica y empírica
begin
    n = 5 # probar n ∈ {5, 20, 100}
    Xmodel = Gamma(2.0, 3.0)
    xobs = rand(Xmodel, n)
    x = collect(range(0.0, quantile(Xmodel, 0.995), length = 1_000))
    yFX = cdf.(Xmodel, x) # Distribución teórica
    yFn = Fn(x, xobs) # Distribución empírica
    yPoli = poligonal(x, xobs) 
    plot(x, yFX, label = "Teórica", legend = :right, lw = 4.0)
    xaxis!("x"); yaxis!("F(x)"); title!("Función de distribución")
    scatter!(x, yFn, label = "Empírica", lw = 3.0, color = :green)
    plot!(x, yPoli, label = "Poligonal", lw = 2.0, color = :pink)
end

# Aproximación Bernstein de cuantiles
begin
    u = collect(range(0.001, 0.999, length = 1_000))
    c = Bernstein_cuantiles(u, xobs)
    plot!(c, u, label = "Bernstein", lw = 2.0, color = :red)
end


# Aproximación de la inversa de cuantiles de Bernstein

begin
    xobs = randn(10_000)
    x = [-1.96, 0, 1.96]
    Bernstein(x, xobs)
end


# Velocidad de cálculo de cuantiles

begin
    n = 50
    xobs = randn(n)
    u = collect(range(0.001, 0.999, length = 999))
end;

@btime poligonal_cuantiles(u, xobs);

@btime Bernstein_cuantiles(u, xobs);

begin
    p = poligonal_cuantiles(u, xobs)
    b = Bernstein_cuantiles(u, xobs)
    scatter(p, b, legend = false, title = "QQ-plot", xlabel = "Poligonal", ylabel = "Bernstein")
    plot!([min(p[1], b[1]), max(p[end], b[end])], [min(p[1], b[1]), max(p[end], b[end])], color = :red, lw = 2.0)
end
