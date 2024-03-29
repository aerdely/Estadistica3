### Comparación de pruebas de bondad de ajuste

# Se simularán muestras a partir de una distribución t-Student con parámetro
# ν ∈ {1,...,30} y se aplicarán pruebas de bondad de ajuste para la hipótesis 
# de que las muestras provienen de una distribución Normal(0,1). Como la
# hipótesis es falsa, las pruebas de bondad de ajuste debieran arrojar
# p-values bajos, mucho má cercanos a cero que a 1, sin embargo conforme
# el parámetro ν incrementa de 1 hacia 30 la distribución t-Student(ν)
# se parece cada vez más a una Normal(0,1) y esto provoca que los p-values
# se alejen cada vez más de cero. Un prueba es uniformemente mejor que otra
# si en este caso sus p-values siempre son más bajos que los de la otra.

begin
    using Distributions, Statistics, Plots, DelimitedFiles
    include("00Stats3.jl")
end

## Construir tablas

function simulación(ν, n, tipo::String, m = 10_000, nsprueba = 100)
    modelo = Normal(0,1) # Ho 
    Fo(x) = cdf(modelo, x)
    simFo(n) = rand(modelo, n)
    pvsims = zeros(m)
    for i ∈ 1:m
        xobs = rand(TDist(ν), n)
        pvsims[i] = GoF(xobs, Fo, simFo; prueba = tipo, numsims = nsprueba, msg = false)
    end
    pvalue = [minimum(pvsims), quantile(pvsims, 0.25), median(pvsims),
              mean(pvsims), quantile(pvsims, 0.75), maximum(pvsims)
    ]
    return pvalue
end

function tabla(n, tipo::String, m = 10_000, nsprueba = 100)
    matriz = zeros(30, 6)
    for ν ∈ 1:30
        matriz[ν, :] = simulación(ν, n, tipo, m, nsprueba)
    end
    encabezados = ["v" "minimo" "1er cuartil" "mediana" "promedio" "3er cuartil" "maximo"]
    tablapv = vcat(encabezados, hcat(collect(1:30), matriz))
    return tablapv
end

@time begin
    m = 10_000
    KS_50 = tabla(50, "KS", m)
    CM_50 = tabla(50, "CM", m)
    AD_50 = tabla(50, "AD", m)
    KS_1000 = tabla(1000, "KS", m)
    CM_1000 = tabla(1000, "CM", m)
    AD_1000 = tabla(1000, "AD", m)
end; # 2 horas aprox

begin
    tablas = ["KS", "CM", "AD"] .* "_50" ∪ ["KS", "CM", "AD"] .* "_1000"
    for t ∈ tablas
        matriz = eval(Meta.parse(t))
        # escribirCSV(matriz, t)
        writedlm(t * ".csv", matriz, ',')
    end
end


## Comparaciones gráficas de ν versus p-value promedio

# Caso 1: Misma prueba, n = 50 versus n = 1,000

function comparación1(prueba::String)
    if prueba == "KS"
        tabla1 = KS_50
        tabla2 = KS_1000
        título = "Kolmogorov - Smirnov"
    elseif prueba == "CM"
        tabla1 = CM_50
        tabla2 = CM_1000
        título = "Cramér - von Mises"
    elseif prueba == "AD"
        tabla1 = AD_50
        tabla2 = AD_1000
        título = "Anderson - Darling"
    else
        return nothing
    end
    ν = tabla1[2:31, 1]
    p1 = tabla1[2:31, 5]
    p2 = tabla2[2:31, 5]
    plot(ν, p1, lw = 3, color = :red, label = "n = 50", legend = (0.8, 0.3))
    plot!(ν, p2, lw = 3, color = :blue, label = "n = 1,000")
    title!(título)
    xaxis!("ν")
    yaxis!("p-value")
    gráfica = current()
    return gráfica
end

for prueba ∈ ["KS", "CM", "AD"]
    gráfica = comparación1(prueba)
    savefig(prueba * ".png")
end

# Caso 2: Las 3 pruebas con n = 50 y con n = 1,000

begin
    ν = KS_50[2:31, 1]
    # n = 50
    plot(ν, KS_50[2:31, 5], lw = 3, color = :red, label = "Kolmogorov - Smirnov", legend = (0.7, 0.3))
    title!("n = 50")
    xaxis!("ν")
    yaxis!("p-value")
    plot!(ν, CM_50[2:31, 5], lw = 3, color = :blue, label = "Cramér - von Mises")
    plot!(ν, AD_50[2:31, 5], lw = 3, color = :green, label = "Anderson - Darling")
    savefig("GoF_50.png")
    # n = 1,000
    plot(ν, KS_1000[2:31, 5], lw = 3, color = :red, label = "Kolmogorov - Smirnov", legend = (0.7, 0.3))
    title!("n = 1,000")
    xaxis!("ν")
    yaxis!("p-value")
    plot!(ν, CM_1000[2:31, 5], lw = 3, color = :blue, label = "Cramér - von Mises")
    plot!(ν, AD_1000[2:31, 5], lw = 3, color = :green, label = "Anderson - Darling")
    savefig("GoF_1000.png")
end
