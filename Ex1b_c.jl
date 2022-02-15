## Solve the Consumption Savings Model using Value Function Iteration
using Plots
include("./Functions/VFI_update_spline.jl")
include("./Functions/TransitionMat.jl")
include("./Functions/broyden.jl")
## 1. Define parameters

# Numerical parameters
# `nk`: Number of points on the asset grid
# `nz`: Number of points on the log-productivity grid
# `crit`: Numerical precision
# `maxk`: Maximum assets
# `mink`: Minimum assets (equal to borrowing limit)
mpar = (nk = 100, nz = 2, crit = 1.0e-5, maxk = 6, mink = -9/4)
println("Numerical parameters")
println(mpar) # Display numerical parameters

# Economic Parameters
# `r`: Real Rate
# `γ`: Coefficient of relative risk aversion
# `β`: Discount factor
# `b`: borrowing limit
par = (r = 4/90, γ = 1.0, β = 0.95, b = mpar.mink)
println("Economic parameters")
println(par) # Display economic parameters

## 2. Generate grids, Meshes and Income
#Define asset grid on log-linearspaced
gri   = (
            k = exp.(collect(range(log(1),log(mpar.maxk-mpar.mink+1);length = mpar.nk))) .- 1 .+ mpar.mink,
            z = [1/9; 10/9]
        )
Π     = [3/5 2/5; 4/90 86/90] # Transition matrix for income


## 3. Define utility functions

if par.γ == 1.0
    util(c)     = log.(c) # Utility
    mutil(c)    = 1.0 ./c  # Marginal utility
    invmutil(mu) = 1.0 ./mu # inverse marginal utility
else
    util(c)      = 1.0/(1.0 - par.γ) .* c .^(1-par.γ) # Utility
    mutil(c)     = 1.0 ./(c .^par.γ) # Marginal utility
    invmutil(mu) = 1.0/(mu .^(1.0 ./par.γ)) # inverse marginal utility
end


## 4a. Value Function Iteration (off-grid)
# Meshes of capital and productivity
meshes = (
            k = [k for k in gri.k, z in gri.z],
            z = [z for k in gri.k, z in gri.z]
         )
Y = meshes.z + meshes.k*(1+par.r) # Cash at hand (Labor income plus assets cum dividend)
timer = time() # Start timer
V    = zeros(mpar.nk,mpar.nz) # Initialize Value Function
distVF = ones(1) # Initialize Distance
iterVF = 1 # Initialize Iteration count
while distVF[iterVF]>mpar.crit # Value Function iteration loop: until distance is smaller than crit.
    # Update Value Function using off-grid search
    global Vnew,kprime = VFI_update_spline(V,Y,util,par,mpar,gri,Π) # Optimize given cont' value
    local dd          = maximum(abs.(Vnew-V)) # Calculate distance between old guess and update
    global V          = Vnew # Update Value Function
    global iterVF     += 1 # Count iterations
    append!(distVF,dd)   # Save distance
end
V      = reshape(V,mpar.nk,mpar.nz)
time1 = time() - timer # Save Time used for VFI

## b) Broyden's method as improvement
timer = time()  # Initialize the timer
Dist_V(V)      = (V - VFI_update_spline(V,Y,util,par,mpar,gri,Π)[1][:]); # Function for the difference of V to its update (exchanging the order matters for speed here!)
V              = zeros(mpar.nk,mpar.nz); # Initialize Value Function
V,x,xx,dist_B  = broyden(Dist_V,V,mpar.crit,1e-14,250); # Finding the root
V,kprime       = VFI_update_spline(V,Y,util,par,mpar,gri,Π) # Update value function, to extract the policy function
V              = reshape(V,mpar.nk,mpar.nz); # Reshaping the value function we get back
time2 = time() - timer

## 4c. Policy Function Iteration (off-grid)
# Meshes of capital and productivity
timer = time() # Start timer
V    = zeros(mpar.nk,mpar.nz) # Initialize Value Function
distPF = ones(1) # Initialize Distance
iterPF = 1
while distPF[iterPF]>mpar.crit # Value Function iteration loop: until distance is smaller than crit.
    # Update Value Function using off-grid search
    global Vnew,kprime = VFI_update_spline(V,Y,util,par,mpar,gri,Π) # Optimize given cont' value
    local U          = util(Y-kprime); # Payoff under optimal policy
    local Gamma      = TransitionMat(kprime,gri,mpar,Π); # Transition matrix
    global Vnew       = (I - par.β*Gamma)\U[:];
    local dd          = maximum(abs.(Vnew[:]-V[:]))
    global V          = Vnew; # Update Value Function
    global iterPF     += 1; #Count iterations
    append!(distPF,dd)
end
V      = reshape(V,mpar.nk,mpar.nz)
time3 = time() - timer # Save Time used for PFI

## 6. Display time and convergence stats
println("Time for solution")
println("VFI:",time1)
println("Broyden ",time2)
println("PFI ",time3)

figure3 = plot(1:length(distVF)-1,distVF[2:end],yaxis = :log,label="VFI")
plot!(1:length(dist_B)-1,dist_B[2:end],label="Broyden")
plot!(1:length(distPF)-1,distPF[2:end],label="PFI")
title!("distance after iterations")