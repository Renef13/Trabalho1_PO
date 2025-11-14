"""
Problema da Dieta - Métodos Heurísticos
Implementação de: Greedy, Busca Local, Relaxação Linear e Arredondamento
"""

import pandas as pd
import numpy as np
from pyomo.environ import (ConcreteModel, Set, Var, Objective, Constraint, ConstraintList,
                           SolverFactory, NonNegativeReals, minimize, value)
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple

# Configuração do matplotlib para exibir gráficos
plt.style.use('default')

class DietaProblem:
    """Classe para resolver o problema da dieta usando diferentes métodos"""

    def __init__(self):
        # Carregar dados dos arquivos CSV
        try:
            self.df_alimentos = pd.read_csv('data/alimentos.csv')
            self.df_restricoes = pd.read_csv('data/restricoes_alimentos.csv')
        except:
            # Dados de backup caso os arquivos não existam
            self.criar_dados_backup()

        self.preparar_dados()

    def criar_dados_backup(self):
        """Cria dados de backup caso os arquivos CSV não existam"""
        self.df_alimentos = pd.DataFrame({
            'Alimento': ['arroz', 'feijao', 'frango', 'leite', 'maca'],
            'Custo_por_100g': [1.0, 1.8, 7.0, 3.5, 2.5],
            'Proteina': [2.5, 8.0, 25.0, 3.4, 0.3],
            'Carboidrato': [28.0, 20.0, 0.0, 5.0, 14.0],
            'Vitamina': [0.1, 1.5, 0.2, 1.2, 2.0]
        })

        self.df_restricoes = pd.DataFrame({
            'Nutriente': ['proteina', 'carboidrato', 'vitamina'],
            'Requisito_Minimo': [70.0, 250.0, 40.0]
        })

    def preparar_dados(self):
        """Prepara os dados para uso nos algoritmos"""
        self.alimentos = self.df_alimentos['Alimento'].tolist()
        self.custos = dict(zip(self.df_alimentos['Alimento'], self.df_alimentos['Custo_por_100g']))

        # Nutrientes (converter para minúsculo para padronizar)
        self.nutrientes = [n.lower() for n in self.df_restricoes['Nutriente'].tolist()]
        self.requisitos = dict(zip([n.lower() for n in self.df_restricoes['Nutriente']],
                                   self.df_restricoes['Requisito_Minimo']))

        # Valores nutricionais
        self.valores_nutricionais = {}
        for _, row in self.df_alimentos.iterrows():
            alimento = row['Alimento']
            self.valores_nutricionais[alimento] = {
                'proteina': row['Proteina'],
                'carboidrato': row['Carboidrato'],
                'vitamina': row['Vitamina']
            }

    def verificar_viabilidade(self, solucao: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """Verifica se uma solução atende às restrições"""
        nutrientes_obtidos = {n: 0.0 for n in self.nutrientes}

        for alimento, quantidade in solucao.items():
            for nutriente in self.nutrientes:
                nutrientes_obtidos[nutriente] += quantidade * self.valores_nutricionais[alimento][nutriente]

        viavel = all(nutrientes_obtidos[n] >= self.requisitos[n] for n in self.nutrientes)
        return viavel, nutrientes_obtidos

    def calcular_custo(self, solucao: Dict[str, float]) -> float:
        """Calcula o custo total de uma solução"""
        return sum(solucao[alimento] * self.custos[alimento] for alimento in self.alimentos)

    def metodo_guloso(self) -> Tuple[Dict[str, float], float, float]:
        """
        Método Guloso (Greedy): Seleciona alimentos com melhor relação custo-benefício
        Estratégia: Prioriza alimentos com maior valor nutricional total por custo
        """
        print("\n=== MÉTODO GULOSO (GREEDY) ===")
        inicio = time.time()

        # Calcular eficiência de cada alimento (nutrientes totais / custo)
        eficiencias = {}
        for alimento in self.alimentos:
            total_nutrientes = sum(self.valores_nutricionais[alimento][n] for n in self.nutrientes)
            eficiencias[alimento] = total_nutrientes / self.custos[alimento]

        # Ordenar alimentos por eficiência (decrescente)
        alimentos_ordenados = sorted(self.alimentos, key=lambda a: eficiencias[a], reverse=True)

        # Construir solução gulosa
        solucao = {a: 0.0 for a in self.alimentos}
        nutrientes_atuais = {n: 0.0 for n in self.nutrientes}

        # Adicionar alimentos até satisfazer restrições
        passo = 0.5  # Incremento em porções de 100g
        max_iteracoes = 1000
        iteracao = 0

        while iteracao < max_iteracoes:
            nutrientes_faltantes = {n: max(0, self.requisitos[n] - nutrientes_atuais[n])
                                   for n in self.nutrientes}

            if all(v == 0 for v in nutrientes_faltantes.values()):
                break

            # Selecionar melhor alimento para o nutriente mais deficiente
            nutriente_critico = max(nutrientes_faltantes.items(), key=lambda x: x[1])[0]

            melhor_alimento = None
            melhor_razao = float('-inf')

            for alimento in alimentos_ordenados:
                valor_nutriente = self.valores_nutricionais[alimento][nutriente_critico]
                if valor_nutriente > 0:
                    razao = valor_nutriente / self.custos[alimento]
                    if razao > melhor_razao:
                        melhor_razao = razao
                        melhor_alimento = alimento

            if melhor_alimento is None:
                break

            # Adicionar porção do melhor alimento
            solucao[melhor_alimento] += passo
            for nutriente in self.nutrientes:
                nutrientes_atuais[nutriente] += passo * self.valores_nutricionais[melhor_alimento][nutriente]

            iteracao += 1

        custo = self.calcular_custo(solucao)
        tempo = time.time() - inicio

        viavel, nutrientes_finais = self.verificar_viabilidade(solucao)
        print(f"Solução viável: {viavel}")
        print(f"Custo total: R$ {custo:.2f}")
        print(f"Tempo de execução: {tempo:.4f}s")
        print("\nQuantidades (porções de 100g):")
        for alimento, qtd in solucao.items():
            if qtd > 0:
                print(f"  {alimento}: {qtd:.2f}")

        return solucao, custo, tempo

    def metodo_busca_local(self, solucao_inicial: Dict[str, float] = None) -> Tuple[Dict[str, float], float, float]:
        """
        Busca Local: Melhora iterativamente uma solução inicial
        Estratégia: Reduz quantidades mantendo viabilidade
        """
        print("\n=== MÉTODO BUSCA LOCAL ===")
        inicio = time.time()

        # Se não há solução inicial, usar método guloso
        if solucao_inicial is None:
            solucao_inicial, _, _ = self.metodo_guloso()

        solucao = solucao_inicial.copy()
        custo_atual = self.calcular_custo(solucao)

        melhorou = True
        iteracoes = 0
        max_iteracoes = 100
        delta = 0.1  # Decremento por iteração

        while melhorou and iteracoes < max_iteracoes:
            melhorou = False
            iteracoes += 1

            # Tentar reduzir cada alimento
            for alimento in self.alimentos:
                if solucao[alimento] > delta:
                    # Tentar reduzir
                    solucao_teste = solucao.copy()
                    solucao_teste[alimento] -= delta

                    viavel, _ = self.verificar_viabilidade(solucao_teste)

                    if viavel:
                        custo_teste = self.calcular_custo(solucao_teste)
                        if custo_teste < custo_atual:
                            solucao = solucao_teste
                            custo_atual = custo_teste
                            melhorou = True

        tempo = time.time() - inicio

        viavel, nutrientes_finais = self.verificar_viabilidade(solucao)
        print(f"Solução viável: {viavel}")
        print(f"Custo total: R$ {custo_atual:.2f}")
        print(f"Tempo de execução: {tempo:.4f}s")
        print(f"Iterações: {iteracoes}")
        print("\nQuantidades (porções de 100g):")
        for alimento, qtd in solucao.items():
            if qtd > 0:
                print(f"  {alimento}: {qtd:.2f}")

        return solucao, custo_atual, tempo

    def metodo_relaxacao_linear(self) -> Tuple[Dict[str, float], float, float]:
        """
        Relaxação Linear: Resolve o problema como LP (variáveis contínuas)
        """
        print("\n=== MÉTODO RELAXAÇÃO LINEAR ===")
        inicio = time.time()

        # Criar modelo Pyomo
        modelo = ConcreteModel()

        # Variáveis de decisão (contínuas, não-negativas)
        modelo.alimentos = Set(initialize=self.alimentos)
        modelo.x = Var(modelo.alimentos, domain=NonNegativeReals)

        # Função objetivo: minimizar custo
        modelo.custo_total = Objective(
            expr=sum(self.custos[a] * modelo.x[a] for a in modelo.alimentos),
            sense=minimize
        )

        # Restrições de nutrientes
        modelo.restricoes_nutrientes = ConstraintList()
        for nutriente in self.nutrientes:
            expr = sum(self.valores_nutricionais[a][nutriente] * modelo.x[a]
                      for a in modelo.alimentos) >= self.requisitos[nutriente]
            modelo.restricoes_nutrientes.add(expr)

        # Resolver
        solver = SolverFactory('glpk')
        resultado = solver.solve(modelo, tee=False)

        # Extrair solução
        solucao = {a: value(modelo.x[a]) for a in self.alimentos}
        custo = value(modelo.custo_total)
        tempo = time.time() - inicio

        viavel, nutrientes_finais = self.verificar_viabilidade(solucao)
        print(f"Solução viável: {viavel}")
        print(f"Custo total: R$ {custo:.2f}")
        print(f"Tempo de execução: {tempo:.4f}s")
        print("\nQuantidades (porções de 100g):")
        for alimento, qtd in solucao.items():
            if qtd > 0.01:
                print(f"  {alimento}: {qtd:.2f}")

        return solucao, custo, tempo

    def metodo_arredondamento(self) -> Tuple[Dict[str, float], float, float]:
        """
        Arredondamento: Resolve LP e arredonda para valores práticos
        Estratégia: Arredonda para cima para manter viabilidade
        """
        print("\n=== MÉTODO ARREDONDAMENTO ===")
        inicio = time.time()

        # Primeiro, resolver relaxação linear
        solucao_lp, _, _ = self.metodo_relaxacao_linear()

        # Arredondar para múltiplos de 0.5 (meio porção)
        solucao_arredondada = {}
        for alimento, valor in solucao_lp.items():
            # Arredondar para cima para manter viabilidade
            if valor > 0:
                solucao_arredondada[alimento] = np.ceil(valor * 2) / 2  # Arredondar para 0.5
            else:
                solucao_arredondada[alimento] = 0.0

        # Verificar viabilidade e ajustar se necessário
        viavel, nutrientes = self.verificar_viabilidade(solucao_arredondada)

        if not viavel:
            # Se não viável, adicionar mais alimentos
            for nutriente in self.nutrientes:
                while nutrientes[nutriente] < self.requisitos[nutriente]:
                    # Encontrar melhor alimento para esse nutriente
                    melhor_alimento = max(self.alimentos,
                                        key=lambda a: self.valores_nutricionais[a][nutriente] / self.custos[a])
                    solucao_arredondada[melhor_alimento] += 0.5
                    nutrientes[nutriente] += 0.5 * self.valores_nutricionais[melhor_alimento][nutriente]

        custo = self.calcular_custo(solucao_arredondada)
        tempo = time.time() - inicio

        viavel, nutrientes_finais = self.verificar_viabilidade(solucao_arredondada)
        print(f"Solução viável: {viavel}")
        print(f"Custo total: R$ {custo:.2f}")
        print(f"Tempo de execução: {tempo:.4f}s")
        print("\nQuantidades (porções de 100g):")
        for alimento, qtd in solucao_arredondada.items():
            if qtd > 0:
                print(f"  {alimento}: {qtd:.2f}")

        return solucao_arredondada, custo, tempo

    def comparar_metodos(self):
        """Executa todos os métodos e compara resultados"""
        print("="*60)
        print("COMPARAÇÃO DE MÉTODOS HEURÍSTICOS - PROBLEMA DA DIETA")
        print("="*60)

        resultados = {}

        # Executar métodos
        sol_guloso, custo_guloso, tempo_guloso = self.metodo_guloso()
        resultados['Guloso'] = {'solucao': sol_guloso, 'custo': custo_guloso, 'tempo': tempo_guloso}

        sol_busca, custo_busca, tempo_busca = self.metodo_busca_local(sol_guloso)
        resultados['Busca Local'] = {'solucao': sol_busca, 'custo': custo_busca, 'tempo': tempo_busca}

        sol_relaxacao, custo_relaxacao, tempo_relaxacao = self.metodo_relaxacao_linear()
        resultados['Relaxação Linear'] = {'solucao': sol_relaxacao, 'custo': custo_relaxacao, 'tempo': tempo_relaxacao}

        sol_arredondamento, custo_arredondamento, tempo_arredondamento = self.metodo_arredondamento()
        resultados['Arredondamento'] = {'solucao': sol_arredondamento, 'custo': custo_arredondamento, 'tempo': tempo_arredondamento}

        # Criar gráficos comparativos
        self.plotar_comparacao(resultados)

        return resultados

    def plotar_comparacao(self, resultados: Dict):
        """Cria gráficos comparativos dos métodos"""
        metodos = list(resultados.keys())
        custos = [resultados[m]['custo'] for m in metodos]
        tempos = [resultados[m]['tempo'] for m in metodos]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparação de Métodos Heurísticos - Problema da Dieta', fontsize=16, fontweight='bold')

        # Gráfico 1: Custo por método
        ax1 = axes[0, 0]
        cores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        bars1 = ax1.bar(metodos, custos, color=cores, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Custo Total (R$)', fontsize=12)
        ax1.set_title('Custo Total por Método', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Adicionar valores nas barras
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'R$ {height:.2f}',
                    ha='center', va='bottom', fontsize=10)

        # Gráfico 2: Tempo de execução
        ax2 = axes[0, 1]
        bars2 = ax2.bar(metodos, tempos, color=cores, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Tempo (segundos)', fontsize=12)
        ax2.set_title('Tempo de Execução por Método', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}s',
                    ha='center', va='bottom', fontsize=9)

        # Gráfico 3: Comparação de soluções (quantidades de alimentos)
        ax3 = axes[1, 0]
        alimentos = self.alimentos
        x = np.arange(len(alimentos))
        width = 0.2

        for i, metodo in enumerate(metodos):
            quantidades = [resultados[metodo]['solucao'][a] for a in alimentos]
            ax3.bar(x + i*width, quantidades, width, label=metodo, alpha=0.8)

        ax3.set_ylabel('Quantidade (porções de 100g)', fontsize=12)
        ax3.set_xlabel('Alimentos', fontsize=12)
        ax3.set_title('Quantidades de Alimentos por Método', fontsize=13, fontweight='bold')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(alimentos, rotation=45, ha='right')
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)

        # Gráfico 4: Comparação normalizada (custo relativo ao melhor)
        ax4 = axes[1, 1]
        custo_minimo = min(custos)
        custos_relativos = [(c / custo_minimo - 1) * 100 for c in custos]

        bars4 = ax4.bar(metodos, custos_relativos, color=cores, alpha=0.8, edgecolor='black')
        ax4.set_ylabel('Desvio do Ótimo (%)', fontsize=12)
        ax4.set_title('Qualidade da Solução (% acima do melhor)', fontsize=13, fontweight='bold')
        ax4.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Melhor solução')
        ax4.grid(axis='y', alpha=0.3)
        ax4.legend()

        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

        plt.tight_layout()
        plt.savefig('comparacao_dieta_metodos.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráficos salvos em 'comparacao_dieta_metodos.png'")
        plt.show()

        # Tabela resumo
        print("\n" + "="*60)
        print("RESUMO COMPARATIVO")
        print("="*60)
        df_resumo = pd.DataFrame({
            'Método': metodos,
            'Custo (R$)': [f'{c:.2f}' for c in custos],
            'Tempo (s)': [f'{t:.4f}' for t in tempos],
            'Desvio (%)': [f'{d:.1f}' for d in custos_relativos]
        })
        print(df_resumo.to_string(index=False))
        print("="*60)


def main():
    """Função principal"""
    print("Inicializando problema da dieta...")

    problema = DietaProblem()
    resultados = problema.comparar_metodos()

    print("\n✓ Análise completa!")


if __name__ == "__main__":
    main()

