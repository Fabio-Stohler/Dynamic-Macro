include("EGM.jl")
include("Young.jl")

"""
    Fsys_Reiter(Xhat, XhatPrime, XSS, mpar, par, idSS,idDev,gri,meshes,Π)

Compute deviations from equilibrium conditions

    * `Xhat`,`XhatPrime`: deviations of present and future variables from steady state
    * `XSS`: steady state values (in logs)
    * `mpar`,`par`: parameter structures
    * `idSS`,`idDev`: index structures for steady state and deviation values
    * `gri`,`meshes`: grids and meshes for income (z) and capital (k)
    * `Π`: transition probability matrix of income (z)

Returns

    * `F`: vector of deviations from equilibrium conditions
"""
function Fsys_Reiter(Xhat, XhatPrime, XSS, mpar, par, idSS,idDev,gri,meshes,Π)
    # expand X around steady state
    
    X      = exp.(Xhat + XSS)
    XPrime = exp.(XhatPrime+ XSS)
    
    # Getting the distribution of k
    Dist            = zeros(eltype(Xhat),mpar.nk, mpar.nz)
    Dist[1:end-1]   = Xhat[idDev.Dist] + XSS[idSS.Dist]
    Dist[end]       = 1 - sum(Dist[:])
    Dist            = Dist[:]
    
    # Getting the distribution of k'
    DistPrime            = zeros(eltype(Xhat),mpar.nk, mpar.nz)
    DistPrime[1:end-1]   = XhatPrime[idDev.Dist] + XSS[idSS.Dist]
    DistPrime[end]       = 1 - sum(DistPrime[:])
    DistPrime            = DistPrime[:]
    #--------------------------------------------------------------------------------------------------------------
    # Aggregate Model
    #--------------------------------------------------------------------------------------------------------------
    F = zeros(eltype(Xhat),length(Xhat))
    # Capital summary
    F[idDev.K] = log(X[idSS.K]) - log(dot(Dist,[gri.k;gri.k]))   # Grid und distribution
    # Output
    F[idDev.Y] = log(X[idSS.Y]) - log(X[idSS.Z]*par.N^(1-par.α)*X[idSS.K]^par.α)  # Output
    # TFP
    F[idDev.Z] = log(XPrime[idSS.Z]) - par.ρ_Z*log(X[idSS.Z])   # TFP law of motion
    # Interest rates
    F[idDev.R] = log(X[idSS.R]) - log(par.α*X[idSS.Y]/X[idSS.K] + (1-par.δ))    
    # Consumption
    F[idDev.C] = log(X[idSS.C]) - log(X[idSS.Y] - X[idSS.I])   # Euler equation, c = u'^(-1)(R*β*E[u'(c')]) 
    # Investment
    F[idDev.I] = log(X[idSS.I]) - log(XPrime[idSS.K] - (1-par.δ)*X[idSS.K])
    
    #--------------------------------------------------------------------------------------------------------------
    # Individual Consumption Policies
    #--------------------------------------------------------------------------------------------------------------
    mutil(c)    = 1.0 ./c  # Marginal utility
    invmutil(mu) = 1.0 ./mu # inverse marginal utility
    if par.γ != 1.0
        mutil(c)     = 1.0 ./(c .^par.γ) # Marginal utility
        invmutil(mu) = 1.0 ./(mu .^(1.0 ./par.γ)) # inverse marginal utility
    end
    w        = X[idSS.Z]*par.N^(-par.α)*X[idSS.K]^par.α*(1-par.α)
    @set! meshes.z = meshes.z*w
    cpol,kprime = EGM(XPrime[idSS.cpol],mutil,invmutil,X[idSS.R],XPrime[idSS.R],par,mpar,Π,meshes,gri)
    
    F[idDev.cpol]= log.(cpol[:]) - log.(X[idSS.cpol])
    
    #--------------------------------------------------------------------------------------------------------------
    # Individual Consumption Policies
    #--------------------------------------------------------------------------------------------------------------
    Γ = YoungSmall(kprime,gri,mpar,Π)
    
    DistNewPrime = Dist'*Γ
    
    F[idDev.Dist] = DistPrime[1:end-1] - DistNewPrime[1:end-1]
    
    return F
end