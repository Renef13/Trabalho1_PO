from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, Binary
import pandas as pd

model = ConcreteModel()

items = ['Notebook Gamer', 'Câmera DSLR', 'Tablet', 'Fone Bluetooth', 'Livro', 'Carregador Portátil']
values = [7500, 4200, 2300, 800, 350, 600]
weights = [2.8, 1.9, 0.7, 0.3, 1.2, 0.4]
capacity = 5.0

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


total_value = sum(values[i] * model.x[items[i]].value for i in range(len(items)))
print(f"Valor total dos itens selecionados: {total_value} R$")
