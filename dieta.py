from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, NonNegativeReals, minimize
import pandas as pd

# Modelo
model = ConcreteModel()

# Alimentos e dados
foods = ['arroz', 'feijao', 'frango', 'leite', 'maçã']
costs = {'arroz': 1.0, 'feijao': 1.8, 'frango': 7.0, 'leite': 3.5, 'maçã': 2.5}  # custo por 100g (R$)
nutrients = ['proteina', 'carboidrato', 'vitamina']
requirements = {'proteina': 120, 'carboidrato': 300, 'vitamina': 60}  # requisitos mínimos (g ou mg equivalentes)
nutrition_values = {
    'arroz': {'proteina': 2.5, 'carboidrato': 28, 'vitamina': 0.1},
    'feijao': {'proteina': 8.0, 'carboidrato': 20, 'vitamina': 1.5},
    'frango': {'proteina': 25, 'carboidrato': 0, 'vitamina': 0.2},
    'leite': {'proteina': 3.4, 'carboidrato': 5, 'vitamina': 1.2},
    'maçã': {'proteina': 0.3, 'carboidrato': 14, 'vitamina': 2.0}
}

# Variáveis de decisão
model.x = Var(foods, within=NonNegativeReals)

# Função objetivo: minimizar custo total
model.obj = Objective(
    expr=sum(costs[f] * model.x[f] for f in foods),
    sense=minimize
)


# Restrições nutricionais
def nutrient_constraints(model, nutrient):
    return sum(nutrition_values[f][nutrient] * model.x[f] for f in foods) >= requirements[nutrient]


model.nutrient_constraints = Constraint(nutrients, rule=nutrient_constraints)

# Resolver o modelo
solver = SolverFactory('glpk')
solver.solve(model)

# Resultados
results_df = pd.DataFrame({
    'Alimento': foods,
    'Quantidade (porções de 100g)': [round(model.x[f].value, 2) for f in foods],
    'Custo Total (R$)': [round(costs[f] * model.x[f].value, 2) for f in foods]
})

results_df['Custo Total (R$)'] = results_df['Custo Total (R$)'].fillna(0)
total_cost = results_df['Custo Total (R$)'].sum()

print("Resultado da Dieta (versão ajustada e realista):")
print(results_df)
print(f"\nCusto total da dieta: R$ {total_cost:.2f}")
