### Estimación no paramétrica de cuantiles

function cuantil_puntual(α, obs)
    sort!(obs)
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

cuantil_puntual(0.5, [1,2,3,4])

using Distributions

α = 0.75
X = Gamma(2, 3)
quantile(X, α) # cuantil teórico

n = 10_000
obs = rand(X, n) # muestra aleatoria
cuantil_puntual(α, obs) # estimación puntual

## Tarea ⇒ programar estimación por intervalo
