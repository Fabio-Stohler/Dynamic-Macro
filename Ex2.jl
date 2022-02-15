## Solve the Consumption Savings Model using an Endogenous Grid Method (EGM)
using Plots
include("./Functions/VFI_update_spline.jl")
include("./Functions/EGM.jl")

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
par = (r = 4/90, γ = 1.0, β = 0.99, b = mpar.mink)
println("Economic parameters")
println(par) # Display economic parameters

## 2. Generate grids, Meshes and Income
#Define asset grid on log-linearspaced
gri   = (
            k = exp.(collect(range(log(1),log(mpar.maxk-mpar.mink+1);length = mpar.nk))) .- 1 .+ mpar.mink,
            #k = collect(range(mpar.mink,mpar.maxk;length = mpar.nk)) # Alternative grid
            z = [1/9; 10/9]
        )
Π     = [3/5 2/5; 4/90 86/90] # Transition matrix for income
# Meshes of capital and productivity
meshes = (
            k = [k for k in gri.k, z in gri.z],
            z = [z for k in gri.k, z in gri.z]
        )
Y = meshes.z + meshes.k*(1+par.r); # Cash at hand (Labor income plus assets cum dividend)

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

## 4. Value Function Iteration

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
timeVFI = time() - timer # Save Time used for VFI

## 5. Plot Policy Functions from VFI

figure1 = plot(gri.k,kprime,labels=["low productivity" "high productivity"],legend = :topleft) # Plot policy functions
plot!(gri.k,gri.k,linestyle = :dash,linecolor= :black,labels = nothing) # Add 45° line
title!("Policy Function from VFI") # Title of the graph
xlabel!("assets")
ylabel!("saving")
savefig("VFI.png")

## 6. Endogenous Grid method using linear interpolation
# $$\frac{\partial u}{\partial c}\left[C^*(k',z)\right]=(1+r) \beta E_{z}\left\{\frac{\partialu}{\partial
# c}\left[C(k',z')\right]\right\}$$

timer = time() # Reset timer
C       = (Y  .- (1+par.r)*gri.k) #Initial guess for consumption policy: roll over assets
Cold    = C # Save old policy
distEG  = ones(1) # Initialize Distance
iterEG  = 1 # Initialize Iteration count
while distEG[iterEG]>mpar.crit
    global C,    = EGM(C, mutil,invmutil,par,mpar,Π,meshes,gri) # Update consumption policy by EGM
    local dd     = maximum(abs.(C-Cold)) # Calculate Distance

    global Cold   = C # Replace old policy
    global iterEG += 1 #count iterations
    append!(distEG,dd)
end
C,Kprimestar = EGM(C,mutil,invmutil,par,mpar,Π,meshes,gri)
timeEGM        = time() - timer #Time to solve using EGM

## 7. Plot Policy Functions from Collocation and compare to VFI

figure2 = plot(gri.k,Kprimestar,labels=["low productivity" "high productivity"],legend= :topleft) #Plot Policy Functions from Collocation
plot!(gri.k,gri.k,linestyle= :dash, linecolor = :black, labels=nothing) # Add 45° line
title!("Policy Function from EGM") # Title of the graph
xlabel!("assets")
ylabel!("saving")
savefig("PFI.png")

figure3 = plot(gri.k,Kprimestar - kprime,labels=["low productivity" "high productivity"],legend= :topright) #Plot Differences in Policies
title!("Difference in Policy Function")

## 8. Compare times of algorithms
println("\nTime to solve (VFI, EGM) ", [timeVFI timeEGM])
println("Iterations to solve (VFI, EGM) ", [iterVF iterEG])