### Nonparametric quantile estimation


## Dependencies

# Packages 

begin
    # internal package (does not requiere installation)
    using Statistics
    # external packages (require previous installation)
    using Distributions
end

# cumulative collection of functions for the course

include("00Stats3.jl")

Stats3()


## Examples

cuantil(0.5, [1,2,3,4])

# the same as `quantile` from the `Statistics` package:
quantile([1,2,3,4], 0.5)


α = 0.75
X = Gamma(2, 3)
quantile(X, α) # theoretical quantil

n = 10_000
obs = rand(X, n) # random sample
cuantil(α, obs)  # point estimation
quantile(obs, α) # point estimation
