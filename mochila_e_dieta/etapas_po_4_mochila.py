import pandas as pd
import numpy as np
from pyomo.environ import (ConcreteModel, Set, Var, Objective, Constraint, ConstraintList,
                           SolverFactory, NonNegativeReals, Binary, maximize, minimize, value)
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple

plt.style.use('default')


class MochilaProblem:

    def __init__(self):
        try:
            self.df_itens = pd.read_csv('../data/itens_mochila.csv')
        except:
            self.criar_dados_backup()

        self.preparar_dados()

    def criar_dados_backup(self):
        self.df_itens = pd.DataFrame({
            'Item': ['Notebook Gamer', 'Camera DSLR', 'Tablet', 'Fone Bluetooth', 'Livro', 'Carregador Portatil'],
            'Valor': [7500, 4200, 2300, 800, 350, 600],
            'Peso': [2.8, 1.9, 0.7, 0.3, 1.2, 0.4]
        })
        self.capacidade = 5.0

    def preparar_dados(self):
        self.itens = self.df_itens['Item'].tolist()
        self.valores = dict(zip(self.df_itens['Item'], self.df_itens['Valor']))
        self.pesos = dict(zip(self.df_itens['Item'], self.df_itens['Peso']))

        if not hasattr(self, 'capacidade'):
            self.capacidade = 5.0

        self.razao_valor_peso = {
            item: self.valores[item] / self.pesos[item]
            for item in self.itens
        }

    def verificar_viabilidade(self, solucao: Dict[str, int]) -> Tuple[bool, float]:

        peso_total = sum(solucao[item] * self.pesos[item] for item in self.itens)
        return peso_total <= self.capacidade, peso_total

    def calcular_valor(self, solucao: Dict[str, int]) -> float:
        return sum(solucao[item] * self.valores[item] for item in self.itens)

    def metodo_guloso(self) -> Tuple[Dict[str, int], float, float]:

        print("\n=== MÉTODO GULOSO (GREEDY) ===")
        inicio = time.time()

        itens_ordenados = sorted(self.itens,
                                 key=lambda item: self.razao_valor_peso[item],
                                 reverse=True)

        solucao = {item: 0 for item in self.itens}
        peso_atual = 0.0

        for item in itens_ordenados:
            # Tentar adicionar o item
            if peso_atual + self.pesos[item] <= self.capacidade:
                solucao[item] = 1
                peso_atual += self.pesos[item]

        valor = self.calcular_valor(solucao)
        tempo = time.time() - inicio

        viavel, peso_total = self.verificar_viabilidade(solucao)
        print(f"Solução viável: {viavel}")
        print(f"Valor total: R$ {valor:.2f}")
        print(f"Peso utilizado: {peso_total:.2f} kg / {self.capacidade:.2f} kg")
        print(f"Tempo de execução: {tempo:.4f}s")
        print("\nItens selecionados:")
        for item, selecionado in solucao.items():
            if selecionado:
                print(f"  ✓ {item} (Valor: R$ {self.valores[item]:.2f}, Peso: {self.pesos[item]:.2f} kg)")

        return solucao, valor, tempo

    def metodo_busca_local(self, solucao_inicial: Dict[str, int] = None) -> Tuple[Dict[str, int], float, float]:

        print("\n=== MÉTODO BUSCA LOCAL ===")
        inicio = time.time()

        if solucao_inicial is None:
            solucao_inicial, _, _ = self.metodo_guloso()

        solucao = solucao_inicial.copy()
        valor_atual = self.calcular_valor(solucao)

        melhorou = True
        iteracoes = 0
        max_iteracoes = 100

        while melhorou and iteracoes < max_iteracoes:
            melhorou = False
            iteracoes += 1

            for item in self.itens:
                if solucao[item] == 0:
                    solucao_teste = solucao.copy()
                    solucao_teste[item] = 1

                    viavel, _ = self.verificar_viabilidade(solucao_teste)

                    if viavel:
                        valor_teste = self.calcular_valor(solucao_teste)
                        if valor_teste > valor_atual:
                            solucao = solucao_teste
                            valor_atual = valor_teste
                            melhorou = True
                            break

            if melhorou:
                continue

            for item_out in self.itens:
                if solucao[item_out] == 1:
                    for item_in in self.itens:
                        if solucao[item_in] == 0:
                            solucao_teste = solucao.copy()
                            solucao_teste[item_out] = 0
                            solucao_teste[item_in] = 1

                            viavel, _ = self.verificar_viabilidade(solucao_teste)

                            if viavel:
                                valor_teste = self.calcular_valor(solucao_teste)
                                if valor_teste > valor_atual:
                                    solucao = solucao_teste
                                    valor_atual = valor_teste
                                    melhorou = True
                                    break

                    if melhorou:
                        break

        tempo = time.time() - inicio

        viavel, peso_total = self.verificar_viabilidade(solucao)
        print(f"Solução viável: {viavel}")
        print(f"Valor total: R$ {valor_atual:.2f}")
        print(f"Peso utilizado: {peso_total:.2f} kg / {self.capacidade:.2f} kg")
        print(f"Tempo de execução: {tempo:.4f}s")
        print(f"Iterações: {iteracoes}")
        print("\nItens selecionados:")
        for item, selecionado in solucao.items():
            if selecionado:
                print(f"  ✓ {item} (Valor: R$ {self.valores[item]:.2f}, Peso: {self.pesos[item]:.2f} kg)")

        return solucao, valor_atual, tempo

    def metodo_relaxacao_linear(self) -> Tuple[Dict[str, float], float, float]:

        print("\n=== MÉTODO RELAXAÇÃO LINEAR ===")
        inicio = time.time()

        modelo = ConcreteModel()

        modelo.itens = Set(initialize=self.itens)
        modelo.x = Var(modelo.itens, domain=NonNegativeReals, bounds=(0, 1))

        modelo.valor_total = Objective(
            expr=sum(self.valores[item] * modelo.x[item] for item in self.itens),
            sense=maximize
        )

        modelo.restricao_peso = Constraint(
            expr=sum(self.pesos[item] * modelo.x[item] for item in self.itens) <= self.capacidade
        )

        solver = SolverFactory('glpk')
        resultado = solver.solve(modelo, tee=False)

        solucao = {item: value(modelo.x[item]) for item in self.itens}
        valor = value(modelo.valor_total)
        tempo = time.time() - inicio

        peso_total = sum(solucao[item] * self.pesos[item] for item in self.itens)

        print(f"Valor total (relaxado): R$ {valor:.2f}")
        print(f"Peso utilizado: {peso_total:.2f} kg / {self.capacidade:.2f} kg")
        print(f"Tempo de execução: {tempo:.4f}s")
        print("\nItens selecionados (frações permitidas):")
        for item, fracao in solucao.items():
            if fracao > 0.01:
                print(
                    f"  {item}: {fracao * 100:.1f}% (Valor: R$ {self.valores[item] * fracao:.2f}, Peso: {self.pesos[item] * fracao:.2f} kg)")

        return solucao, valor, tempo

    def metodo_arredondamento(self) -> Tuple[Dict[str, int], float, float]:

        print("\n=== MÉTODO ARREDONDAMENTO ===")
        inicio = time.time()

        solucao_lp, _, _ = self.metodo_relaxacao_linear()

        itens_ordenados = sorted(self.itens,
                                 key=lambda item: solucao_lp[item],
                                 reverse=True)

        solucao = {item: 0 for item in self.itens}
        peso_atual = 0.0

        for item in itens_ordenados:
            if solucao_lp[item] >= 0.5:
                if peso_atual + self.pesos[item] <= self.capacidade:
                    solucao[item] = 1
                    peso_atual += self.pesos[item]
            elif solucao_lp[item] > 0:
                if peso_atual + self.pesos[item] <= self.capacidade:
                    espaco_restante = self.capacidade - peso_atual
                    if self.pesos[item] <= espaco_restante:
                        solucao[item] = 1
                        peso_atual += self.pesos[item]

        valor = self.calcular_valor(solucao)
        tempo_total = time.time() - inicio

        viavel, peso_total = self.verificar_viabilidade(solucao)
        print(f"Solução viável: {viavel}")
        print(f"Valor total: R$ {valor:.2f}")
        print(f"Peso utilizado: {peso_total:.2f} kg / {self.capacidade:.2f} kg")
        print(f"Tempo de execução: {tempo_total:.4f}s")
        print("\nItens selecionados:")
        for item, selecionado in solucao.items():
            if selecionado:
                print(f"  ✓ {item} (Valor: R$ {self.valores[item]:.2f}, Peso: {self.pesos[item]:.2f} kg)")

        return solucao, valor, tempo_total

    def metodo_branch_and_bound_simples(self) -> Tuple[Dict[str, int], float, float]:
        print("\n=== MÉTODO BRANCH AND BOUND (ÓTIMO) ===")
        inicio = time.time()

        modelo = ConcreteModel()

        modelo.itens = Set(initialize=self.itens)
        modelo.x = Var(modelo.itens, domain=Binary)

        modelo.valor_total = Objective(
            expr=sum(self.valores[item] * modelo.x[item] for item in self.itens),
            sense=maximize
        )

        modelo.restricao_peso = Constraint(
            expr=sum(self.pesos[item] * modelo.x[item] for item in self.itens) <= self.capacidade
        )

        solver = SolverFactory('glpk')
        resultado = solver.solve(modelo, tee=False)

        solucao = {item: int(value(modelo.x[item])) for item in self.itens}
        valor = value(modelo.valor_total)
        tempo = time.time() - inicio

        viavel, peso_total = self.verificar_viabilidade(solucao)
        print(f"Solução ótima: {viavel}")
        print(f"Valor total: R$ {valor:.2f}")
        print(f"Peso utilizado: {peso_total:.2f} kg / {self.capacidade:.2f} kg")
        print(f"Tempo de execução: {tempo:.4f}s")
        print("\nItens selecionados:")
        for item, selecionado in solucao.items():
            if selecionado:
                print(f"  ✓ {item} (Valor: R$ {self.valores[item]:.2f}, Peso: {self.pesos[item]:.2f} kg)")

        return solucao, valor, tempo

    def comparar_metodos(self):
        print("=" * 60)
        print("COMPARAÇÃO DE MÉTODOS HEURÍSTICOS - PROBLEMA DA MOCHILA")
        print("=" * 60)

        resultados = {}

        sol_guloso, valor_guloso, tempo_guloso = self.metodo_guloso()
        resultados['Guloso'] = {'solucao': sol_guloso, 'valor': valor_guloso, 'tempo': tempo_guloso}

        sol_busca, valor_busca, tempo_busca = self.metodo_busca_local(sol_guloso)
        resultados['Busca Local'] = {'solucao': sol_busca, 'valor': valor_busca, 'tempo': tempo_busca}

        sol_relaxacao, valor_relaxacao, tempo_relaxacao = self.metodo_relaxacao_linear()
        resultados['Relaxação Linear'] = {'solucao': sol_relaxacao, 'valor': valor_relaxacao, 'tempo': tempo_relaxacao}

        sol_arredondamento, valor_arredondamento, tempo_arredondamento = self.metodo_arredondamento()
        resultados['Arredondamento'] = {'solucao': sol_arredondamento, 'valor': valor_arredondamento,
                                        'tempo': tempo_arredondamento}

        sol_otimo, valor_otimo, tempo_otimo = self.metodo_branch_and_bound_simples()
        resultados['Ótimo (B&B)'] = {'solucao': sol_otimo, 'valor': valor_otimo, 'tempo': tempo_otimo}

        self.plotar_comparacao(resultados)

        return resultados

    def plotar_comparacao(self, resultados: Dict):
        metodos = list(resultados.keys())
        valores = [resultados[m]['valor'] for m in metodos]
        tempos = [resultados[m]['tempo'] for m in metodos]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparação de Métodos Heurísticos - Problema da Mochila', fontsize=16, fontweight='bold')

        ax1 = axes[0, 0]
        cores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        bars1 = ax1.bar(metodos, valores, color=cores, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Valor Total (R$)', fontsize=12)
        ax1.set_title('Valor Total por Método', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'R$ {height:.0f}',
                     ha='center', va='bottom', fontsize=9)

        ax2 = axes[0, 1]
        bars2 = ax2.bar(metodos, tempos, color=cores, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Tempo (segundos)', fontsize=12)
        ax2.set_title('Tempo de Execução por Método', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}s',
                     ha='center', va='bottom', fontsize=9)

        ax3 = axes[1, 0]

        itens_display = [item[:15] + '...' if len(item) > 15 else item for item in self.itens]

        for i, metodo in enumerate(metodos[:-1]):
            if metodo == 'Relaxação Linear':
                continue
            solucao = resultados[metodo]['solucao']
            y_pos = [j + i * 0.15 for j in range(len(self.itens))]
            selecao = [solucao[item] if isinstance(solucao[item], int) else 0 for item in self.itens]
            ax3.barh(y_pos, selecao, height=0.15, label=metodo, alpha=0.8)

        ax3.set_yticks(range(len(self.itens)))
        ax3.set_yticklabels(itens_display, fontsize=9)
        ax3.set_xlabel('Selecionado (1 = Sim, 0 = Não)', fontsize=12)
        ax3.set_title('Itens Selecionados por Método', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(axis='x', alpha=0.3)

        ax4 = axes[1, 1]
        valor_otimo = valores[-1]  # Último é o ótimo
        qualidade = [(v / valor_otimo) * 100 for v in valores]

        bars4 = ax4.bar(metodos, qualidade, color=cores, alpha=0.8, edgecolor='black')
        ax4.set_ylabel('Qualidade (% do Ótimo)', fontsize=12)
        ax4.set_title('Qualidade da Solução', fontsize=13, fontweight='bold')
        ax4.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Ótimo')
        ax4.grid(axis='y', alpha=0.3)
        ax4.legend()
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('comparacao_mochila_metodos.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráficos salvos em 'comparacao_mochila_metodos.png'")
        plt.show()

        print("\n" + "=" * 70)
        print("RESUMO COMPARATIVO")
        print("=" * 70)
        df_resumo = pd.DataFrame({
            'Método': metodos,
            'Valor (R$)': [f'{v:.2f}' for v in valores],
            'Tempo (s)': [f'{t:.4f}' for t in tempos],
            'Qualidade (%)': [f'{q:.1f}' for q in qualidade]
        })
        print(df_resumo.to_string(index=False))
        print("=" * 70)


def main():
    print("Inicializando problema da mochila...")

    problema = MochilaProblem()
    resultados = problema.comparar_metodos()

    print("\n✓ Análise completa!")


if __name__ == "__main__":
    main()
