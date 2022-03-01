### Pruebas de Bondad de Ajuste (Goodness of fit tests)

begin
    using Distributions, Statistics, Plots 
    include("00Stats3.jl")
end


# Simulemos una muestra aleatoria tamaño 100 a partir de
# una distribución de probabilidad Normal(0,1)
begin
    X = Normal(0,1)
    xobs = rand(X, 100)
    mean(xobs), var(xobs)
end

# Ahora supongamos no sabemos de qué distribución de probabilidad
# proviene la muestra, y realicemos pruebas de bondad de ajuste
# para decidir si se rechaza o no que proviene de una distribución
# de probabilidad t-Student con parámetro ν = 2. Sabemos que esta
# hipótesis debiera ser rechazada, por lo que esperamos p-valores
# de las pruebas mucho más cercanos a cero que a 1.

begin
    x = range(-4, 4, length = 1000)
    plot(x, pdf.(Normal(0, 1), x), lw = 3, color = :green, label = "Normal(0,1)")
    ν = 2
    plot!(x, pdf.(TDist(ν), x), lw = 3, color = :red, label = "t-Student ($ν)")
end

begin
    modelo = TDist(ν)
    Fo(x) = cdf(modelo, x)
    simFo(n) = rand(modelo, n)
    GoF(xobs, Fo, simFo) # Anderson-Darling por default
end


GoF(xobs, Fo, simFo; prueba = "CM") # Cramér - von Mises

GoF(xobs, Fo, simFo; prueba = "KS") # Kolmogorov - Smirnov


# Ahora hagamos pruebas de bondad de ajuste la distribución correcta.
# Esperamos valores más cercanos a 1 que a cero.

begin
    modelo = Normal(0,1)
    Fo(x) = cdf(modelo, x)
    simFo(n) = rand(modelo, n)
    GoF(xobs, Fo, simFo) # Anderson-Darling por default
end

GoF(xobs, Fo, simFo; prueba = "CM") # Cramér - von Mises

GoF(xobs, Fo, simFo; prueba = "KS") # Kolmogorov - Smirnov
