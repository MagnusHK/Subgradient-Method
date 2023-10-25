# Implementation of a simple subgradient method for solving the Lagrangian Dual


# Import packages
using JuMP, GLPK, LinearAlgebra, Plots


# Knapsack problem
# Z = max cx
# s.t. Dx <= d
# x in {0,1}^3

c = [11.0, 5.0, 14.0]
D = [3.0 2.0 4.0]
d = 5.0

# Iterations
n = 100

# Initialize the Lagrange multipliers
u = ones(1,n)

# Initialize vectors for storage
z_u = zeros(1,n)

# Define 0 < pi < 2
pi = 1

# Define the upper bound solution
z_ub = 1000 # to be found

# Define model
model = Model(GLPK.Optimizer)

@variable(model, x[1:3], Bin)

for t in 1:(n-1)

    @objective(model, Min, sum(c[i]*x[i] for i=1:3) + u[t]*(d-sum(D[i]*x[i] for i=1:3)))

    # Solve Lagrangian subproblem IP(u(t))
    JuMP.optimize!(model)

    # Solution
    xsol = JuMP.value.(x)
    z_u[t] = JuMP.objective_value(model)

    # Compute the current violation of the 'complicated' constraints
    s = d-(D*xsol)[1]

    # Compute T
    T = pi*(z_ub - z_u[t])/s

    # Iterate u
    u[t+1] = max(0,u[t]+T*s)
    
end





# Plot
plot(u,z_u)

t = range(1,100, length=100)
plot(t, u)


