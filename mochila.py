from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, Binary
import pandas as pd

model = ConcreteModel()

items = ['item1', 'item2', 'item3', 'item4', 'item5']
values = [5000, 300, 2000, 1500, 50]  # Valores dos itens
weights = [2.5, 0.8, 0.2, 0.6, 0.4]  # Pesos dos itens
capacity = 3.0  # Capacidade da mochila

model.x = Var(items, domain=Binary)

model.obj = Objective(expr=sum(values[i] * model.x[items[i]] for i in range(len(items))), sense='maximize')

model.capacity_constraint = Constraint(expr=sum(weights[i] * model.x[items[i]] for i in range(len(items))) <= capacity)

# o modelo
solver = SolverFactory('glpk')
solver.solve(model)


results = pd.DataFrame({
    'Item': items,
    'Valor': values,
    'Peso': weights,
    'Selecionado': [model.x[item].value for item in items]
})

print("Resultado da Mochila 0-1:")
print(results)
