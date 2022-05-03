### Algoritmo de bisección par encontrar raíces de funciones continuas

begin
    using Plots, BenchmarkTools
    include("00Stats3.jl")
end


begin
    f(x) = (x - 3) * (x - 1) * (x + 1)
    xx = range(-2, 4, length = 1_000)
    plot(xx, f.(xx), lw = 2, legend = false, xlabel = "x", ylabel = "f(x)")
    hline!([0], color = :orange)
end


r1 = biseccion(f, -1.9, 0)

r1.numiter

f(r1.raíz) # comprobando

r2 = biseccion(f, 0.5, 1.4)

r3 = biseccion(f, 2.9, 3.2)

begin
    r = [r1.raíz, r2.raíz, r3.raíz]
    scatter!(r, f.(r), color = :red)
end


# Si violamos el supuesto de que exista una raíz única en el intervalo $[a,b]$
# entonces dependiendo de la elección de los valores de $a$ y $b$ puede
# encontrarse cualquiera de ellas:

biseccion(f, -1.9, 3.5)

biseccion(f, -1.1, 3.1)

biseccion(f, -1.1, 3.5)


# Y si violamos el supuesto de exista raíz en el intervalo $[a,b]$ entonces
# se obtendrá un resultado inaceptable:

biseccion(f, -2, -1.5)


# ¿Qué pasa si buscamos la raíz mediante *fuerza bruta*?

A = [3, 4, 1, 7, 1]

findmin(A) # y existe su contraparte `findmax`

findmin(A)[1]

findall(A .== findmin(A)[1]) # todas las posiciones donde hay mínimos globales

function fuerzaBruta(f, a, b; δ = abs((a + b)/2) / 1_000_000)
    n = Int(round(ceil((b - a) / δ)))
    z = collect(range(a, b, length = n))
    v = abs.(f.(z))
    r = z[findmin(v)[2]]
    return (raíz = r, dif = f(r), tol = δ)
end

fuerzaBruta(f, -1.9, 0)

biseccion(f, -1.9, 0)

@btime fuerzaBruta(f, -1.9, 0)

@btime biseccion(f, -1.9, 0)

function fuerzaBruta2(f, a, b; δ = abs((a + b)/2) / 1_000_000)
    n = Int(round(ceil((b - a) / δ)))
    Δ = (b - a) / n
    r = a 
    v = abs(f(r))
    for i ∈ 1:n
        r2 = a + i*Δ
        v2 = abs(f(r2))
        if v2 < v
            r = r2
            v = v2
        end
    end
    return (raíz = r, dif = f(r), tol = Δ)
end

fuerzaBruta2(f, -1.9, 0)

@btime fuerzaBruta2(f, -1.9, 0)
