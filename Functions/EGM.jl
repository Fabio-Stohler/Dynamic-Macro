using CubicSplines
include("LinInterpol.jl")
"""
    EGM(C,mutil,invmutil,par,mpar,Π,meshes,gri)

Iterate forward the consumption policies for the consumption savings model using the EGM method. 

    * `C` (k x z) is the consumption policy guess.
    * `mutil` and `invmutil` are the marginal utility functions and its inverse.
    * `par` and `mpar` are parameter structures.
    * `Π` is the transition probability matrix.
    * `meshes` and `gri` are meshes and grids for income (z) and assets (k).

Returns

    * `C`: (k x z) consumption policy
    * `Kprime`: (k x z) asset policy
"""
function EGM(C,mutil,invmutil,par,mpar,Π,meshes,gri)
    mu     = mutil(C) # Calculate marginal utility from c'
    emu    = mu*Π'    # Calculate expected marginal utility
    Cstar  = invmutil((1+par.r)*par.β.*emu)     # Calculate cstar(m',z)
    Kstar  = (meshes.k  .+ Cstar .- meshes.z)/(1+par.r) # Calculate mstar(m',z)
    Kprime = copy(meshes.k) # initialize Capital Policy

    for z=1:mpar.nz # For massive problems, this can be done in parallel
        # generate savings function k(z,kstar(k',z))=k'
        Savings     = CubicSpline(sort(Kstar[:,z]),gri.k[sortperm(Kstar[:,z])])
        # Implement linear extrapolation
        mink = minimum(Kstar[:,z]); maxk = maximum(Kstar[:,z])
        minSlope = (Savings[mink+0.01]-Savings[mink])/0.01  # Numerical derivative at mink
        maxSlope = (Savings[maxk]-Savings[maxk-0.01])/0.01  # Numerical derivative at maxk
        function SavingsExtr(k)
            # Function that, given k, gives back savings. Given k is outside the grid, we extrapolate the functional values
            if mink <= k <= maxk
                return Savings[k] 
            elseif k < mink
                return Savings[mink]+minSlope*(k-mink)  # Extrapolating
            else
                return Savings[maxk]+maxSlope*(k-maxk)  # Extrapolating
            end
        end 
        Kprime[:,z] = SavingsExtr.(gri.k)   # Obtain k'(z,k) by interpolation
        BC          = gri.k .< Kstar[1,z]   # Check Borrowing Constraint
        # Replace Savings for HH saving at BC
        Kprime[BC,z].= mpar.mink # Households with the BC flag choose borrowing contraint
    end
    # generate consumption function c(z,k^*(z,k'))
    C          = meshes.z + (1+par.r)*meshes.k - Kprime # Consumption update (not using Kstar, since this is the old policy function)
    return C, Kprime
end

"""
    EGM(C,mutil,invmutil,R,RPrime,par,mpar,Π,meshes,gri)

Iterate forward the consumption policies for the consumption savings model using the EGM method. 

    * `C` (k x z) is the consumption policy guess.
    * `mutil` and `invmutil` are the marginal utility functions and its inverse.
    * `R` and `RPrime`: gross interest rates this and next period
    * `par` and `mpar` are parameter structures.
    * `Π` is the transition probability matrix.
    * `meshes` and `gri` are meshes and grids for income (z) and assets (k).

Returns

    * `C`: (k x z) consumption policy
    * `Kprime`: (k x z) asset policy
"""
function EGM(C,mutil,invmutil,R,RPrime,par,mpar,Π,meshes,gri)
    C      = reshape(xxx,mpar.nk,mpar.nz)
    mu     = mutil(xxx) # Calculate marginal utility from c'
    emu    = xxx     # Calculate expected marginal utility
    Cstar  = invmutil(xxx)     # Calculate cstar(m',z)
    Kstar  = (xxx  + xxx - xxx)/RPrime # Calculate mstar(m',z)
    Kprime = ones(eltype(xxx),size(meshes.k)) # initialize Capital Policy

    for z=1:mpar.nz # For massive problems, this can be done in parallel
        # generate savings function k(z,kstar(k',z))=k'
        Kprime[:,z]  = mylinearinterpolate(sort(xxx),gri.k[sortperm(xxx)],gri.k) # Obtain k'(z,k) by interpolation
        BC          = xxx .< xxx # Check Borrowing Constraint
        # Replace Savings for HH saving at BC
        Kprime[BC,z].= xxx # Households with the BC flag choose borrowing contraint
    end
    # generate consumption function c(z,k^*(z,k'))
    C          = xxx + xxx - xxx #Consumption update
    return C, Kprime
end
    