from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, NonNegativeReals, minimize
import pandas as pd

model = ConcreteModel()

foods = ['feijao']
costs = [1.2]
nutrients = ['proteina', 'carboidrato', 'vitamina']
requirements = {'proteina': 50, 'carboidrato': 80, 'vitamina': 20}
nutrition_values = {'feijao': [5, 20, 10]}

model.x = Var(foods, within=NonNegativeReals)

model.obj = Objective(expr=sum(costs[i] * model.x[foods[i]] for i in range(len(foods))), sense=minimize)


def nutrient_constraints(model, nutrient):
    index = nutrients.index(nutrient)
    return sum(nutrition_values[food][index] * model.x[food] for food in foods) >= requirements[nutrient]


model.nutrient_constraints = Constraint(nutrients, rule=nutrient_constraints)

solver = SolverFactory('glpk')
results = solver.solve(model)

results_df = pd.DataFrame({
    'Alimento': foods,
    'Quantidade (porções)': [model.x[food].value for food in foods]
})

print("Resultado da Dieta:")
print(results_df)

