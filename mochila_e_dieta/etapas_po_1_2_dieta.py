from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, NonNegativeReals, minimize, value
import pandas as pd

model = ConcreteModel()

# Alimentos e dados
foods = ['arroz', 'feijao', 'frango', 'leite', 'maçã']
costs = {'arroz': 1.0, 'feijao': 1.8, 'frango': 7.0, 'leite': 3.5, 'maçã': 2.5}  # custo por 100g (R$)
nutrients = ['proteina', 'carboidrato', 'vitamina']
requirements = {'proteina': 70, 'carboidrato': 250, 'vitamina': 40}  # requisitos mínimos (g ou mg equivalentes)
nutrition_values = {
    'arroz': {'proteina': 2.5, 'carboidrato': 28, 'vitamina': 0.1},
    'feijao': {'proteina': 8.0, 'carboidrato': 20, 'vitamina': 1.5},
    'frango': {'proteina': 25, 'carboidrato': 0, 'vitamina': 0.2},
    'leite': {'proteina': 3.4, 'carboidrato': 5, 'vitamina': 1.2},
    'maçã': {'proteina': 0.3, 'carboidrato': 14, 'vitamina': 2.0}
}

# Limites mínimos (porções de 100g)
min_limits = {'arroz': 0.5, 'feijao': 0.5, 'frango': 0.5, 'leite': 0.3, 'maçã': 0.3}

model.x = Var(
    foods,
    within=NonNegativeReals,
    bounds=lambda m, f: (min_limits[f], None)
)

model.obj = Objective(
    expr=sum(costs[f] * model.x[f] for f in foods),
    sense=minimize
)


def nutrient_constraints(model, nutrient):
    return sum(nutrition_values[f][nutrient] * model.x[f] for f in foods) >= requirements[nutrient]


model.nutrient_constraints = Constraint(nutrients, rule=nutrient_constraints)

solver = SolverFactory('glpk')
solver.solve(model)

results_df = pd.DataFrame({
    'Alimento': foods,
    'Quantidade (porções de 100g)': [round(value(model.x[f]), 2) for f in foods],
    'Custo Total (R$)': [round(costs[f] * value(model.x[f]), 2) for f in foods]
})

total_cost = results_df['Custo Total (R$)'].sum()

print("\nResultado da Dieta:")
print(results_df)
print(f"\nCusto total da dieta: R$ {total_cost:.2f}")

for n in nutrients:
    total_n = sum(nutrition_values[f][n] * value(model.x[f]) for f in foods)
    print(f"{n}: {total_n:.2f} (mínimo {requirements[n]})")

# PS: como nao coloquei limites maximos, o modelo pode sugerir quantidades muito altas de certos
# alimentos para minimizar o custo, desde que os requisitos nutricionais sejam atendidos.
