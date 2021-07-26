starttime = time()
# Specific parameter adjustments
@set! mpar.maxk = 20    # Maximimum assets
@set! mpar.mink = -9/4    # Minimum Assets (equal to Borrowing Limit)
@set! par.b     = mpar.mink # Borrowing Limit
@set! par.γ = 4

println("--------------------------")
println("Exercise 5: Aiyagari model")
println("--------------------------")
println(par) # Display economic parameters

## 1. Generate grids, Meshes and Income
@set! gri.k   = exp.(collect(range(log(1),log(mpar.maxk-mpar.mink+1);length = mpar.nk))) .- 1 .+ mpar.mink # Define asset grid on log-linearspaced
# Meshes of capital and productivity
meshes = (
            k = [k for k in gri.k, z in gri.z],
            z = [z for k in gri.k, z in gri.z]
         )
# Calculate stationary labor supply
aux = Π^1000
N   = dot(aux[1,:],gri.z)

## 2. Calculate Excess demand
Kdemand(R)      = (par.α/(R+par.δ))^(1/(1-par.α))*N   # Calculate capital demand by firms for a given interest rate and employment
rate(K)         = par.α*(K/N)^(par.α-1) - par.δ       # Calculate the return on capital given K and employment N 
wage(K)         = (1-par.α)*(K/N)^(par.α)             # Calculate the wage rate given K and employment N 
ExcessDemand(K) = K_Agg(rate(K),wage(K),par,mpar,Π,meshes,gri)[1] .- K           # Calculate the difference between capital supply and demand for wages and returns given by assumed capital demand


Rgrid = [-0.01:.0025:.045;]   # a grid for interest rates for plotting
KD    = Kdemand.(Rgrid)    # calculate capital demand for these rates
ExD   = ExcessDemand.(KD)  # calculate excess demand for these amounts of capital


## 3. Find equilibrium
Rstar_Aiyagari  = rate(fzero(ExcessDemand,(Kdemand(0.045),Kdemand(0.00)))) # find equilibrium amount of capital (and corresponding rate)
println("Equilibrium Interest Rate: ",Rstar_Aiyagari)
println("Equilibrium Capital Stock: ",Kdemand(Rstar_Aiyagari))
println(time() - starttime)

## 4. Plot
figure4 = plot(ExD+KD,Rgrid,label="Supply of Funds");
plot!(KD,Rgrid,linecolor = :black, label="Demand for funds");
plot!([mpar.mink, mpar.maxk]*0.8,[Rstar_Aiyagari, Rstar_Aiyagari], linecolor = :black, linestyle = :dot, label = "Equilibrium Rate", legend = :bottomright);
xlabel!("Funds");
ylabel!("interest rate");
display(figure4)