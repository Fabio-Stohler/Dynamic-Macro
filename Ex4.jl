
# Specific parameter adjustments
@set! mpar.maxk = 10         # Maximimum assets
@set! mpar.mink = -9/4       # Minimum Assets (equal to Borrowing Limit)
@set! par.b     = mpar.mink  # Borrowing Limit
@set! par.γ = 2
println("-------------------------")
println("Exercise 4: Hugget model")
println("-------------------------")
println(par) # Display economic parameters

## 1. Generate grids, Meshes and Income
@set! gri.k   = exp.(collect(range(log(1),log(mpar.maxk-mpar.mink+1);length = mpar.nk))) .- 1 .+ mpar.mink # Define asset grid on log-linearspaced
# Meshes of capital and productivity
meshes = (
            k = [k for k in gri.k, z in gri.z],
            z = [z for k in gri.k, z in gri.z]
         )
## 2. Calculate Excess demand
ExcessDemand(R)  = K_Agg(R,1,par,mpar,Π,meshes,gri)[1] - 0   # Calculate the difference between capital supply and demand for wages and returns given by assumed capital demand

Rgrid = -0.01:.001:.049                                      # a grid for interest rates for plotting
ExD   = ExcessDemand.(Rgrid)                                     # calculate excess demand for these rates                             
## 3. Find equilibrium
Rstar_Huggett = fzero(ExcessDemand,(-0.00,0.049))               # Calculate equilibrium rate
## 4. Plot
figure3 = plot(ExD,Rgrid,label="Supply of Funds");
plot!(0*Rgrid,Rgrid,linecolor=:black,label="Demand for funds");
plot!([mpar.mink, mpar.maxk]*0.8,[Rstar_Huggett, Rstar_Huggett], linestyle = :dot,linecolor = :black,label="Equilibrium Rate",legend=:bottomright);
xlabel!("Funds");
ylabel!("interest rate");
println("Equilibrium Interest Rate:", Rstar_Huggett);
display(figure3)