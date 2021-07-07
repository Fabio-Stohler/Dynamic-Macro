using Parameters, Plots, Setfield, Interpolations, Random, Roots
## 1. Define parameters (Common header for Exercise 3-5)
cd("C:/Users/fasto/Dropbox/Desktop/Computational Macro/Julia/Dynamic-Macro/")
include("Functions/K_Agg.jl")

@with_kw struct NumericalParameters
    nk = 100 # Number of points on the asset grid
    nz = 2 # Number of points on the log-productivity grid
    crit = 1.0e-5 # Numerical precision
    maxk = 6 # Maximum assets
    mink = -9/4 # Minimum assets (equal to borrowing limit)
    T = 100000
end
mpar = NumericalParameters()

@with_kw struct EconomicParameters
    r = 0 # Real Rate (for Exercise 3)
    γ = 2 # Coefficient of relative risk aversion
    β = 0.95 # Discount factor
    α = 0.36 # Capital Share (for Exercise 5)
    δ = 0.1 # Depreciation rate (for Exercise 5)
    b = mpar.mink # borrowing limit
end
par = EconomicParameters()

## 2. Grids and transitions
Π  = [3/5 2/5; 4/90 86/90]
struct Grids
    k # capital
    z # income
end
gri   = Grids(
            exp.(collect(range(log(1),log(mpar.maxk-mpar.mink+1);length = mpar.nk))) .- 1 .+ mpar.mink,
            [1/9; 10/9]
        )

timer = time()       
## I. Exercise 3: Bewley model
include("Ex3.jl")
## II. Exercise 4: Hugget model
include("Ex4.jl")
## III. Exercise 5: Aiyagari model
include("Ex5.jl")
timeTotal   = time() - timer;
println("Total time for calculation ", timeTotal)