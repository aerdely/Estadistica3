#=
    Inferencia estadística a partir de una muestra aleatoria tamaño n
    proveniente de una distribubión de probabilidad Beta(α,β) con 
    ambos parámetros α > 0 y β > 0 desconocidos.

    Se pretende realizar estimación puntual de los parámetros, así como
    una prueba de hipótesis para  Ho: α > β
=#


### Paquetes a utilizar (deben estar previamente instalados)

using Distributions, StatsPlots, StatsBase, SpecialFunctions, Optim



### Simular muestra aleatoria observada

begin
    n = 50
    α = 0.7 # valor teórico del parámetro
    β = 0.9 # valor teórico del parámetro
    X = Beta(α, β)
    xobs = rand(X, n)
end



### Inferencia clásica

begin # Estimación puntual: método de momentos
    m(x) = mean(x)
    s2(x) = var(x)
    α_mom(x) = (m(x)^2)*(1-m(x))/s2(x) - m(x)
    β_mom(x) = m(x)*((1-m(x))^2)/s2(x) - 1 + m(x)
    estmom = (α_mom(xobs), β_mom(xobs))
    println("Estimación puntual por método de momentos:")
    println("  (α,β) = ", estmom)
end

begin # Estimación puntual: máxima verosimilitud
    t1 = sum(log.(xobs))
    t2 = sum(log.(1 .- xobs))
    logL(z) = n*logbeta(z[1],z[2]) - t1*(z[1]-1) - t2*(z[2]-1)
    EMV = optimize(logL, [α_mom(xobs), β_mom(xobs)])
    estmv = (EMV.minimizer[1], EMV.minimizer[2])
    println("Estimación puntual por máxima verosimilitud:")
    println("  (α,β) = ", estmv)
end

function bootstrap(estadístico, xobs; nB = 10_000, prob = 0.95)
    n = length(xobs)
    T = estadístico
    Tobs = T(xobs)
    dimT = length(Tobs)
    Tsim = zeros(nB, dimT)
    for j ∈ 1:nB
        muestra = rand(xobs, n) # pseudo-muestra bootstrap
        Tsim[j, :] = T(muestra)
    end
    estim_puntual = zeros(dimT)
    estim_intervalo = zeros(dimT, 2)
    for k ∈ 1:dimT
        estim_puntual[k] = mean(Tsim[:, k])
        estim_intervalo[k, :] = quantile(Tsim[:, k], [(1 - prob)/2, (1 + prob)/2])
    end
    estim = (puntual = estim_puntual, intervalo = estim_intervalo, valor = Tobs, sims = Tsim, p = prob)
    return estim
end;

begin
    # estimación puntual y por intervalo vía bootstrap paramétrico
    T(x) = [α_mom(x), β_mom(x)]
    estboot = bootstrap(T, xobs, prob = 0.95)
    println("Estimación vía boostrap paramétrico")
    println("-----------------------------------")
    println("Puntual: (α,β) = ", transpose(estboot.puntual))
    println("Por intervalo de confianza $(100*estboot.p)%") 
    println("  α ∈ ", transpose(estboot.intervalo[1, : ]))
    println("  β ∈ ", transpose(estboot.intervalo[2, :]))
    println("Estimación bootstrap de la probabilidad de Ho: α > β")
    mean(estboot.sims[:, 1] .> estboot.sims[:, 2])
end

begin
    # gráfica simulación bootstrap
    α_boot, β_boot = estboot.sims[:, 1], estboot.sims[:, 2]
    scatter(α_boot, β_boot, size = (400,400), ms = 0.5, mc = :black, label = "simulaciones bootstrap")
    xaxis!("α")
    yaxis!("β")
    scatter!([estboot.puntual[1]], [estboot.puntual[2]], ms = 4.0, mc = :green2, label = "estimación puntual")
    G1 = scatter!([α],[β], ms = 4.0, mc = :cyan, label = "valor correcto")
end

begin
    # histograma bivariado simulación bootstrap
    histogram2d(α_boot, β_boot, size = (400,400), label = "simulaciones bootstrap")
    xaxis!("α")
    yaxis!("β")
    scatter!([estboot.puntual[1]], [estboot.puntual[2]], ms = 4.0, mc = :green2, label = "estimación bootstrap")
    G2 = scatter!([α],[β], ms = 4.0, mc = :cyan, label = "valor correcto")
end



### Inferencia bayesiana

function ABC(Iα, Iβ; nsims = 10_000, nselec = 1_000, prob = 0.95)
    δ(x,y) = √sum((x .- y) .^ 2)
    w1(x) = sum(log.(x))
    w1obs = w1(xobs)
    w2(x) = sum(log.(1 .- x))
    w2obs = w2(xobs)
    αsim = rand(Uniform(Iα[1], Iα[2]), nsims)
    βsim = rand(Uniform(Iβ[1], Iβ[2]), nsims)
    dist = zeros(nsims)
    for i ∈ 1:nsims
        xsim = rand(Beta(αsim[i], βsim[i]), n)
        dist[i] = δ([w1(xsim), w2(xsim)], [w1obs, w2obs])
    end
    iselec = sortperm(dist)[1:nselec]
    αselec = αsim[iselec]
    βselec = βsim[iselec]
    estbayes = (puntual = (median(αselec), median(βselec)),
                αint = quantile(αselec, [(1 - prob)/2, (1 + prob)/2]),
                βint = quantile(βselec, [(1 - prob)/2, (1 + prob)/2]),
                p = prob, sims = hcat(αselec, βselec)
    )
    return estbayes
end;

begin
    eb = ABC([0.0,3.0], [0.0,3.0], nsims = 1_000_000, nselec = 1_000)
    println("Estimación bayesiana (ABC)")
    println("--------------------------")
    println("Puntual: (α,β) = ", eb.puntual)
    println("Por intervalo de probabilidad $(100*eb.p)%") 
    println("  α ∈ ", transpose(eb.αint))
    println("  β ∈ ", transpose(eb.βint))
    println("Estimación bayesiana de la probabilidad de Ho: α > β")
    mean(eb.sims[:, 1] .> eb.sims[:, 2]) 
end

begin
    # gráfica simulación bayesiana
    α_bayes, β_bayes = eb.sims[:, 1], eb.sims[:, 2]
    scatter(α_bayes, β_bayes, size = (400,400), ms = 0.5, mc = :black, label = "simulaciones bayesianas")
    xaxis!("α")
    yaxis!("β")
    scatter!([eb.puntual[1]], [eb.puntual[2]], ms = 4.0, mc = :green2, label = "estimación puntual")
    G3 = scatter!([α],[β], ms = 4.0, mc = :cyan, label = "valor correcto")
end

begin
    # histograma bivariado simulación bayesiana
    histogram2d(α_bayes, β_bayes, size = (400,400), label = "simulaciones bayesianas")
    xaxis!("α")
    yaxis!("β")
    scatter!([eb.puntual[1]], [eb.puntual[2]], ms = 4.0, mc = :green2, label = "estimación bayesiana")
    G4 = scatter!([α],[β], ms = 4.0, mc = :cyan, label = "valor correcto")
end



### Estimación por intervalo de la media

begin
    media_teórica = α/(α+β)
    println("Valor teórico de la media:")
    println("α/(α+β) = ", media_teórica)
end


## Inferencia paramétrica clásica
#= 
    Se asume que la media muestral tiene una distribución aproximadamente
    Normal con media igual a la media muestral y varianza igual a la varianza
    muestral, y a partir de ello se estandariza y obtiene un intervalo de
    confianza 95%
=#
begin
    z = quantile(Normal(0,1), 0.975)
    μ = mean(xobs)
    σ = std(xobs)
    println("Estimación clásica de la media: Intervalo de confianza 95%")
    media_estclasica = (μ - z*σ, μ + z*σ)
end


## Inferencia no paramétrica vía bootstrap
function bootstrap2(estadístico, xobs; nB = 10_000, prob = 0.95)
    n = length(xobs)
    T = estadístico
    Tobs = T(xobs)
    Tsim = zeros(nB)
    for j ∈ 1:nB
        muestra = rand(xobs, n) # pseudo-muestra bootstrap
        Tsim[j] = T(muestra)
    end
    estim_puntual = mean(Tsim)
    estim_intervalo = quantile(Tsim, [(1 - prob)/2, (1 + prob)/2])
    estim = (puntual = estim_puntual, intervalo = estim_intervalo, valor = Tobs, sims = Tsim)
    return estim
end;
begin
    estboot = bootstrap2(mean, xobs, prob = 0.95)
    println("Estimación por intervalo no paramétrica (bootstrap)")
    println(transpose(estboot.intervalo))
end


## Inferencia bayesiana
begin
    prob = 0.95
    μ_bayes = α_bayes ./ (α_bayes .+ β_bayes)
    bayes_int = quantile(μ_bayes, [(1 - prob)/2, (1 + prob)/2])
    println("Estimación bayesiana de intervalo de probabilidad 95%")
    println(transpose(bayes_int))
end

