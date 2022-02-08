### Empirical Distribution Function (edf)


## Dependencies

# Packages 

begin
    # internal package (does not requiere installation)
    using Statistics
    # external packages (require previous installation)
    using Distributions, Plots, LaTeXStrings, BenchmarkTools
end

# cumulative collection of functions for the course

include("00Stats3.jl")

Stats3()


## Simple examples

methods(Fn)
xobs = randn(10_000) # Normal(0,1) random sample
Fn(0.0, xobs) # ≈ 0.5
Fn([-1.96, 0.0, 1.96], xobs)
diff(Fn([-1.96, 1.96], xobs)) # ≈ 0.95


## Benchmarking

# Alternative calculations of Fn

function Gn(x::Real, xobs::Vector{<:Real})
    sum(xobs .≤ x) / length(xobs)
end

function Hn(x::Real, xobs::Vector{<:Real})
    mean(xobs .≤ x) # depends on `mean` function in `Statistics` internal package
end

xobs = randn(10_000_000) # huge random sample

@time Fn(0.0, xobs) # @time is a Julia macro
@time Fn(0.0, xobs) # changes every time

@time Gn(0.0, xobs)
@time Gn(0.0, xobs)

@time Hn(0.0, xobs)
@time Hn(0.0, xobs)

# benchmarks

@btime Fn(0.0, xobs) # @btime is a BenchmarkTools macro, calculates an average
@btime Gn(0.0, xobs) # as fast as Fn
@btime Hn(0.0, xobs) # slower


## Example: compare edf versus theoretical distribution function in a graph

function simFn(probmodel, n, numpoints = 1_000)
    # probmodel = a probability model as in the `Distributions` package
    # n = simulated sample size
    # numpoints = number of points to evaluate de theoretical cdf 
    xobs = rand(probmodel, n)
    x = collect(range(minimum(xobs), maximum(xobs), length = numpoints))
    y = Fn(x, xobs)
    begin
        plot(x, cdf(probmodel, x), lw = 3, label = "model cdf", legend = (.15,.85))
        scatter!(x, y, markersize = 3, label = "observed edf")
        xaxis!(L"x")
        yaxis!(L"F_X(x)")
        title!(string(probmodel))
    end
    return current()
end

simFn(Normal(0, 1), 50)

simFn(Gamma(2, 3), 50)

simFn(Binomial(20, 0.6), 1_000)


## Generalization of the edf
#  P(X ∈ |a,b|) where -∞ ≤ a ≤ b ≤ ∞ and `|` is either `]` or `[`

xobs = randn(1_000_000) # Normal(0,1) simulation

@time Tn("]-1.96,1.96]", xobs) # ≈ 0.95
@time Tn("]-1.96,1.96]", xobs) # ≈ 0.95

@btime Tn("]-1.96,1.96]", xobs) # ≈ 0.95
@btime diff(Fn([-1.96, 1.96], xobs))[1]
@btime Fn(1.96, xobs) - Fn(-1.96, xobs)

X = Gamma(2, 3)
mode(X), median(X), mean(X)
1 - cdf(X, mode(X)) # theoretical value of P(X > mode(X))
rX = rand(X, 1_000_000) # random sample from X 
intervalo = "]$(mode(X)),Inf["
Tn(intervalo, rX)

Y = Binomial(50, 0.6)
mean(Y)
cdf(Y, 29) - cdf(Y, 24) # theoretical calue of P(25 ≤ Y < E(Y))
rY = rand(Y, 1_000_000)
Tn("[25,30[", rY)
