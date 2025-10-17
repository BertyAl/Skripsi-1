import gurobipy as gp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Hidden cell to avoid licensing messages
# when docs are generated.
with gp.Model():
    pass
# Import example data
Sigma = pd.read_pickle("sigma.pkl")
mu = pd.read_pickle("mu.pkl")

mu_bar = 0.5  # Required return

# Create an empty optimization model
m = gp.Model()

# Add variables: x[i] denotes the proportion of capital invested in stock i
# 0 <= x[i] <= 1
x = m.addMVar(len(mu), lb=0, ub=1, name="x")

# Budget constraint: all investments sum up to 1
m.addConstr(x.sum() == 1, name="Budget_Constraint")

# Lower bound on expected return
ret_constr = m.addConstr(mu.to_numpy() @ x >= mu_bar, name="Min_Return")

# Define objective function: Minimize overall risk
m.setObjective(x @ Sigma.to_numpy() @ x, gp.GRB.MINIMIZE)

m.optimize()

print(f"Minimum risk:     {m.ObjVal:.6f}")
print(f"Expected return:  {mu @ x.X:.6f}")
print(f"Solution time:    {m.Runtime:.2f} seconds\n")

# Print investments (with non-negligible value, i.e., > 1e-5)
positions = pd.Series(name="Position", data=x.X, index=mu.index)
print(f"Number of trades: {positions[positions > 1e-5].count()}\n")
print(positions[positions > 1e-5])

returns = np.linspace(0.21, 0.5, 20)
risks = np.zeros(returns.shape)
npos = np.zeros(returns.shape)

# Hide Gurobi log output
m.params.OutputFlag = 0

# Solve the model for each risk level
for i, ret in enumerate(returns):
    # Modify lower bound on expected return
    ret_constr.RHS = ret
    m.optimize()
    # Store data
    risks[i] = np.sqrt(x.X @ Sigma @ x.X)
    npos[i] = len(x.X[x.X > 1e-5])

fig, axs = plt.subplots(1, 2, figsize=(10, 3))

# Axis 0: The efficient frontier
axs[0].scatter(x=risks, y=returns, marker="o", label="sample points", color="Red")
axs[0].plot(risks, returns, label="efficient frontier", color="Red")
axs[0].set_xlabel("Standard deviation")
axs[0].set_ylabel("Expected return")
axs[0].legend()
axs[0].grid()

# Axis 1: The number of open positions
axs[1].scatter(x=returns, y=npos, color="Red")
axs[1].plot(returns, npos, color="Red")
axs[1].set_xlabel("Expected return")
axs[1].set_ylabel("Number of positions")
axs[1].grid()

plt.show()