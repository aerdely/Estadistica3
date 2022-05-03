### Aproximación poligonal de una función de distribución continua

begin
    using Distributions, Plots, LaTeXStrings
    include("00Stats3.jl")
end

## Ejemplo: m.a ~ Gamma(2, 3)

# Simular muestra y calcular distribución teórica y empírica
begin
    n = 5 # probar n ∈ {5, 100, 1_000}
    Xmodel = Gamma(2.0, 3.0)
    xobs = rand(Xmodel, n)
    x = collect(range(0.0, quantile(Xmodel, 0.995), length = 1_000))
    yFX = cdf.(Xmodel, x) # Distribución teórica
    yFn = Fn(x, xobs) # Distribución empírica
end;

# Aproximación poligonal + gráfica comparativa
begin
    yPoligonal = poligonal(x, xobs) # probar parámetros opcionales: mínimo = 0.0, máximo = ...
    plot(x, yFX, label = "Teórica", legend = :right, lw = 4.0)
    xaxis!(L"x"); yaxis!(L"F(x)"); title!("Función de distribución")
    plot!(x, yFn, label = "Empírica", lw = 3.0, color = :green)
    plot!(x, yPoligonal, label = "Poligonal", lw = 2.0, color = :red)
end


# poligonal_densidad -> no es muy buena, por ello no se incluye en 00Stats3.jl
function poligonal_densidad(x::Vector{<:Real}, xobs::Vector{<:Real}; mínimo = minimum(xobs), máximo = maximum(xobs))
    n = length(xobs)
    xord = sort(xobs)
    xp = (xord[1:(n-1)] .+ xord[2:n]) ./ 2
    xp = vcat(mínimo, xp, máximo)
    function g(z, xp)
        v = 0.0
        if xp[1] < z ≤ xp[n+1]
            for k ∈ 2:(n+1)
                v += 1*(xp[k-1] < z ≤ xp[k]) / (xp[k] - xp[k-1])
            end
            v /= n
        end
        return v
    end
    m = length(x)
    polivalores = zeros(m)
    for i ∈ 1:m
        polivalores[i] = g(x[i], xp)
    end
    return polivalores
end

# Simular muestra
begin
    n = 5 # probar n ∈ {5, 100, 1_000}
    Xmodel = Gamma(2.0, 3.0)
    xobs = rand(Xmodel, n)
    x = collect(range(0.0, quantile(Xmodel, 0.995), length = 1_000))
    yFX = pdf.(Xmodel, x) # Densidad teórica
end;

# Densidad poligonal + gráfica comparativa
begin
    yPoligonal = poligonal_densidad(x, xobs) # probar parámetros opcionales: mínimo = 0.0, máximo = ...
    plot(x, yFX, label = "Teórica", legend = :right, lw = 4.0)
    xaxis!(L"x"); yaxis!(L"f(x)"); title!("Función de densidad")
    plot!(x, yPoligonal, label = "Poligonal", lw = 2.0, color = :red)
end


# simular muestra + aproximación poligonal
begin
    n = 5 
    Xmodel = Gamma(2.0, 3.0)
    xobs = rand(Xmodel, n)
    x = collect(range(0.0, quantile(Xmodel, 0.995), length = 1_000))
    yPoligonal = poligonal(x, xobs)
    plot(x, yPoligonal, label = "Poligonal", legend = :right, lw = 4.0, color = :green)
    xaxis!(L"x"); yaxis!(L"F(x)"); title!("Función de distribución")
end

# Aproximación poligonal de cuantiles
begin
    u = collect(range(0.001, 0.999, length = 1_000))
    c = poligonal_cuantiles(u, xobs)
    plot!(c, u, label = "Poligonal inversa", lw = 2.0, color = :red)
end


# Simular muestras a partir de muestras observadas
begin
    n = 1_000 
    Xmodel = Gamma(2.0, 3.0)
    xobs1 = rand(Xmodel, n)
    xobs2 = rand(Xmodel, n)
    u = rand(n)
    xsim1 = poligonal_cuantiles(u, xobs1)
    xsim2 = poligonal_cuantiles(u, xobs2)
    sort!(xobs1); sort!(xobs2)
    sort!(xsim1); sort!(xsim2)
end;

## QQ-plots

# observado1 vs observado2
begin
    scatter(xobs1, xobs2, legend = false, title = "QQ-plot", 
            xlabel = "muestra observada 1", ylabel = "muestra observada 2"
    )
    plot!([0, max(xobs1[end], xobs2[end])], [0, max(xobs1[end], xobs2[end])], color = :red)
end

# observado1 vs simulado1
begin
    scatter(xobs1, xsim1, legend = false, title = "QQ-plot", 
            xlabel = "muestra observada 1", ylabel = "muestra simulada 1"
    )
    plot!([0, max(xobs1[end], xsim1[end])], [0, max(xobs1[end], xsim1[end])], color = :red)
end

# observado2 vs simulado2
begin
    scatter(xobs2, xsim2, legend = false, title = "QQ-plot", 
            xlabel = "muestra observada 2", ylabel = "muestra simulada 2"
    )
    plot!([0, max(xobs2[end], xsim2[end])], [0, max(xobs2[end], xsim2[end])], color = :red)
end

# Pruebas Anderson-Darling

Fo(x) = pdf(Xmodel, x)
simFo(n) = rand(Xmodel, n)
GoF(xobs1, Fo, simFo)
GoF(xsim1, Fo, simFo)
GoF(xobs2, Fo, simFo)
GoF(xsim2, Fo, simFo)
