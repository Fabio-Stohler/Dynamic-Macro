## Exercise 6: A HANC model, Reiter's method
using Parameters, Setfield, PrettyTables, Roots, ForwardDiff, LinearAlgebra, Plots
cd("C:/Users/fasto/Dropbox/Desktop/Computational Macro/Julia/Dynamic-Macro")
include("Functions/K_Agg.jl")
include("Functions/Fsys_Reiter.jl")
include("Functions/Klein.jl")


## 1. Define parameters
@with_kw struct NumericalParameters
    nk = 100 # Number of points on the capital grid
    nz = 2 # Number of points on the employment grid
    crit = 1.0e-10 # Precision up to which to solve the value function
    maxk = 20 # highest point on the capital grid
    mink = 0 # lowest point on the capital grid
    T = 10 # length of IRF
    nstates = 1 # Number of aggregate states 
    ncontrols = 1 # Number of aggregate controls
    ntotal = 2 # Number of parameters totally
end
mpar = NumericalParameters()

@with_kw struct EconomicParameters
    r = 0 # Real Rate
    γ = 4 # Coefficient of relative risk aversion
    β = 0.95 # Discount factor
    α = 0.36 # Curvature of production function
    δ = 0.1 # Depreciation
    ρ_Z = 0.75 # persistence of technology shock
    UIB = 0.5 # Unemployment benefit as fraction of employed
    b = mpar.mink # borrowing limit
    N = 1 # aggregate labor supply for productivity h
    τ = 0
end
par = EconomicParameters()

## Produce grids
Π  = [3/5 2/5; 4/90 86/90]
aux     = Π^1000
E       = aux[1,2]
U       = 1-E
@set! par.N   = E*10/9 # correct aggregate labor supply for productivity h
@set! par.τ = par.UIB *U/E # unemployment budget $$U * UIB *w*h = \tau*E*w*h \implies \tau = UIB *U/E$$
struct Grids
    k # capital
    z # income
end

gri   = Grids(
            exp.(collect(range(log(1),log(mpar.maxk-mpar.mink+1);length = mpar.nk))) .- 1 .+ mpar.mink, # individual capital grid
            [par.UIB*10/9; 10/9 *(1-par.τ)] # to be multiplied by w
        )

# Meshes of capital and productivity
meshes = (
            k = [k for k in gri.k, z in gri.z],
            z = [z for k in gri.k, z in gri.z]
         )

## Display Model
pretty_table(
             hcat([
                 "Discount Factor";
                 "Returns to Scale";
                 "Relative Risk Aversion";
                 "Depreciation";
                 "Unemployment Benefit";
                 "UI-Contributions"
              ],
              [
                  par.β; par.α; par.γ; par.δ; par.UIB; par.τ
              ])
            )

## 2. Solve the incomplete markets model w/o aggregate risk
Kdemand(R)      = par.N * (par.α/(R+par.δ)).^(1/(1-par.α))         # Calculate capital demand by firms for a given interest rate and employment
rate(K)         = par.α* par.N.^(1-par.α) * K.^(par.α-1) -par.δ # Calculate the return on capital given K and employment N 
wage(K)         = (1-par.α)* par.N.^(-par.α) * K.^(par.α)          # Calculate the wage rate given K and employment N 
ExcessDemand(K) = K_Agg(rate(K),wage(K),par,mpar,Π,meshes,gri)[1] - K   # Calculate the difference between capital supply and demand for wages and returns given by assumed capital demand

Rstar           = rate(fzero(ExcessDemand,(Kdemand(0.052),Kdemand(-0.01)))) # find equilibrium amount of capital (and corresponding rate)
Kstar           = Kdemand(Rstar)
~,saving_star,marginal_k, StDist, Γ,c_star = K_Agg(rate(Kstar),wage(Kstar),par,mpar,Π,meshes,gri)


## 3. introduce aggregate Risk: Reiter method
#Create indexes to access entries in steady state and deviation thereof
#vector
count        = 0
# States
struct Indexes
    Dist
    Z
    cpol
    R
    K
    Y
    C
    I
end
idSS = Indexes(zeros(8)...)

# states
@set! idSS.Dist = count .+ (1:(length(StDist)-1))
count += length(idSS.Dist) # leave out one element from distribution

@set! idSS.Z    = count + 1; count += 1
# controls
@set! idSS.cpol = count .+ (1:length(c_star))
count += length(idSS.cpol)
@set! idSS.R    = count + 1; count += 1
@set! idSS.K    = count + 1; count += 1
@set! idSS.Y    = count + 1; count += 1
@set! idSS.C    = count + 1; count += 1
@set! idSS.I    = count + 1; count += 1
idDev       = idSS # for Reiter no difference between deviation and steady state size (no compression)

XSS             = zeros(count)
XSS[idSS.cpol]  = c_star
XSS[idSS.R]     = 1+Rstar
XSS[idSS.K]     = Kstar
XSS[idSS.Y]     = par.N^(1-par.α)*Kstar^par.α
XSS[idSS.C]     = par.N^(1-par.α)*Kstar^par.α - par.δ*Kstar
XSS[idSS.I]     = par.δ*Kstar
XSS[idSS.Z]     = 1
XSS             = log.(XSS)
XSS[idSS.Dist]  = StDist[1:end-1]

@set! mpar.nstates    = length(idDev.Dist) + 1
@set! mpar.ncontrols  = length(XSS)-mpar.nstates
@set! mpar.ntotal     = mpar.nstates + mpar.ncontrols

F(devs) = Fsys_Reiter(devs[1:mpar.ntotal],devs[mpar.ntotal+1:end],XSS,mpar,par,idSS,idDev,gri,meshes,Π)
DF = ForwardDiff.jacobian(F,zeros(2*mpar.ntotal))
A = DF[:,mpar.ntotal+1:end]
B = DF[:,1:mpar.ntotal]

S2C,S2SPrime  = Klein(A,B,mpar)

## 4. Plot IRF
s_t = zeros(mpar.nstates,mpar.T+1)
x_t = zeros(mpar.ntotal,mpar.T)
s_t[idDev.Z,1] = 1
for t=1:mpar.T
    s_t[:,t+1] = S2SPrime*s_t[:,t]
    x_t[:,t] = [I(mpar.nstates) ; S2C] *s_t[:,t]
end

figure1 = plot(x_t[[idDev.Y, idDev.C, idDev.I, idDev.R],:]',labels=["Output" "Consumption" "Investment" "interest rate"],legend= :topright);
xlabel!("Periods")
ylabel!("Impulse Responses")
display(figure1)