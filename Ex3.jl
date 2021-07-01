
# Specific parameter adjustments
@set! mpar.mink = 0
@set! mpar.maxk = 3 # Maximum assets
@set! par.b = mpar.mink
@set! par.γ = 2
println("-------------------------")
println("Exercise 3: Bewley model")
println("-------------------------")
println(par) # Display economic parameters
## 1. Generate grids, Meshes and Income
@set! gri.k   = collect(range(mpar.mink,mpar.maxk;length=mpar.nk))
# Meshes of capital and productivity
meshes = (
            k = [k for k in gri.k, z in gri.z],
            z = [z for k in gri.k, z in gri.z]
         )

## 2. Equilibrium in Bewley model with Young method
K,kprime,marginal_k, = K_Agg(par.r,1,par,mpar,Π,meshes,gri) # Solve for equlibrium

## 3. Plot Policy Functions
figure1 = plot(gri.k,kprime,labels=["low productivity" "high productivity"],legend = :topleft); # Plot policy functions
plot!(gri.k,gri.k,linestyle = :dash,linecolor= :black,labels = nothing); # Add 45° line
title!("Policy Value Functions");
xlabel!("assets");
ylabel!("saving");
display(figure1)

## 4. Simulate the economy
# Define an off-grid savings function by interpolation
timer = time()
Saving = interpolate((gri.k,gri.z),kprime,Gridded(Linear())) # Continuous Savings Function as Linear Interpolant
# Simulate the exogeneous state
Random.seed!(42)
PI      = cumsum(Π,dims = 2)      # Cummulative Transition matrix
ϵ       = rand(mpar.T)            # Random numbers for simulation
S       = rand(1:mpar.nz,mpar.T)  # Starting value
k       = zeros(mpar.T)           # Starting value for NEXT PERIODS ASSETS assets
for t=2:mpar.T
    S[t]    = count(PI[S[t-1],:].< ϵ[t])+1   # Update productivity state
    k[t]    = Saving(k[t-1],gri.z[S[t-1]])     # Update Assest Holdings
end
timeSim   = time() - timer

# Plot histogram
figure2 = histogram(k[10001:end],bins=(gri.k[1:end-1] + gri.k[2:end])/2,normalize= :probability,label="Simulation");
title!("Distribution of asset holdings");

## 5. Compare solutions
bar!(gri.k,marginal_k,label="Direct Calculation",legend= :topleft);
display(figure2)

timer = time() # time Young method 
Γ, StDist = Young(kprime,gri,mpar,Π)
timeYoung = time() - timer

println("")
println("Average Asset holdings")
println("----------------------")
println("Simulation: ",sum(k[10001:end])/length(k[10001:end]))
println("Young     : ",K)
println("----------------------")
println("Time for calculation")
println("--------------------")
println("Simulation: ",timeSim)
println("Young     : ",timeYoung)