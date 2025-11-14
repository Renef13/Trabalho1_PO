"""
etapas_po_4.py

Heurísticas para aproximação de soluções combinatórias no Problema da Mochila 0-1,
utilizando os dados disponíveis em data/C1.csv .. data/C4.csv (e itens_mochila.csv).

Métodos implementados:
 - Gulosa (Greedy) por razão valor/peso
 - Relaxação Linear (Mochila fracionária) via abordagem gulosa fracionária
 - Arredondamento (a partir da relaxação linear) com reparo e complemento guloso
 - Busca Local (1-add e 1-1 swap) partindo de uma solução inicial

Saídas:
 - Relatórios por instância
 - Tabela comparativa com valores, tempos e gaps em relação à relaxação linear

Observação:
 - Desenvolvido de forma independente; apenas inspirado em ideias gerais de
   trabalhos anteriores, porém com código próprio e autocontido.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


# ============================
# Utilidades de carregamento
# ============================

def carregar_instancia_mochila(csv_path: str, capacidade_padrao: float = 15.0) -> Tuple[List[str], np.ndarray, np.ndarray, float]:
    """Carrega uma instância de mochila a partir de um CSV.

    O CSV deve conter colunas: Item, Valor, Peso. Opcionalmente pode conter uma
    linha com Item="Capacidade" e o valor da capacidade na coluna Valor.

    Caso não exista a linha de capacidade, será usado `capacidade_padrao`.
    Linhas vazias ou incompletas serão ignoradas.
    """
    df = pd.read_csv(csv_path)

    # Normaliza nomes de colunas (evita problemas de capitalização)
    cols = {c.lower(): c for c in df.columns}
    for required in ["item", "valor", "peso"]:
        if required not in cols:
            raise ValueError(f"Arquivo {csv_path} deve conter a coluna '{required.capitalize()}'")

    col_item = cols["item"]
    col_val = cols["valor"]
    col_peso = cols["peso"]

    df = df.copy()
    # Detecta a linha de capacidade (quando presente)
    has_capacity = False
    if df[col_item].dtype != object:
        df[col_item] = df[col_item].astype(str)
    mask_cap = df[col_item].str.strip().str.lower() == "capacidade"
    if mask_cap.any():
        cap_row = df.loc[mask_cap].iloc[0]
        capacidade = float(cap_row[col_val])
        has_capacity = True
        # Remove a linha de capacidade da lista de itens
        df = df.loc[~mask_cap]
    else:
        capacidade = float(capacidade_padrao)

    # Remove linhas vazias/incompletas
    df = df.dropna(subset=[col_item, col_val, col_peso])

    items = df[col_item].astype(str).tolist()
    values = df[col_val].astype(float).to_numpy()
    weights = df[col_peso].astype(float).to_numpy()

    return items, values, weights, capacidade


# ============================
# Estruturas e helpers
# ============================

@dataclass
class Solucao:
    selecao: np.ndarray  # vetor 0/1 indicando itens escolhidos
    valor: float
    peso: float
    tempo_ms: float
    metodo: str


def avaliar(selecao: np.ndarray, valores: np.ndarray, pesos: np.ndarray) -> Tuple[float, float]:
    valor = float(np.dot(selecao, valores))
    peso = float(np.dot(selecao, pesos))
    return valor, peso


def melhor_item_que_cabe(valores: np.ndarray, pesos: np.ndarray, capacidade: float) -> Tuple[np.ndarray, float, float]:
    selecao = np.zeros_like(valores, dtype=int)
    melhor_valor = -np.inf
    melhor_idx = -1
    for i in range(len(valores)):
        if pesos[i] <= capacidade and valores[i] > melhor_valor:
            melhor_valor = valores[i]
            melhor_idx = i
    if melhor_idx >= 0:
        selecao[melhor_idx] = 1
    valor, peso = avaliar(selecao, valores, pesos)
    return selecao, valor, peso


# ============================
# Gulosa (Greedy)
# ============================

def gulosa_ratio(valores: np.ndarray, pesos: np.ndarray, capacidade: float) -> Solucao:
    inicio = time.perf_counter()
    n = len(valores)
    ordem = np.argsort(-(valores / np.maximum(pesos, 1e-12)))  # maior razão primeiro
    selecao = np.zeros(n, dtype=int)
    restante = capacidade
    for i in ordem:
        if pesos[i] <= restante:
            selecao[i] = 1
            restante -= pesos[i]
    valor, peso = avaliar(selecao, valores, pesos)
    tempo_ms = (time.perf_counter() - inicio) * 1000.0
    return Solucao(selecao, valor, peso, tempo_ms, metodo="Gulosa")


# ============================
# Relaxação Linear (Mochila Fracionária)
# ============================

def relaxacao_linear(valores: np.ndarray, pesos: np.ndarray, capacidade: float) -> Tuple[np.ndarray, float, float, float]:
    """Resolve a mochila fracionária por razão valor/peso.

    Retorna (x_fracionario, valor_fracionario, peso_usado, tempo_ms).
    """
    inicio = time.perf_counter()
    n = len(valores)
    ordem = np.argsort(-(valores / np.maximum(pesos, 1e-12)))
    x = np.zeros(n, dtype=float)
    restante = capacidade
    valor_total = 0.0
    for i in ordem:
        if restante <= 1e-12:
            break
        if pesos[i] <= restante:
            x[i] = 1.0
            restante -= pesos[i]
            valor_total += valores[i]
        else:
            frac = restante / pesos[i]
            x[i] = frac
            valor_total += frac * valores[i]
            restante = 0.0
            break
    peso_usado = float(np.dot(x, pesos))
    tempo_ms = (time.perf_counter() - inicio) * 1000.0
    return x, float(valor_total), peso_usado, tempo_ms


# ============================
# Arredondamento + Reparo
# ============================

def arredondamento_reparo(
    x_frac: np.ndarray,
    valores: np.ndarray,
    pesos: np.ndarray,
    capacidade: float,
    selecao_gulosa: np.ndarray | None = None,
) -> Solucao:
    inicio = time.perf_counter()

    # 1) Arredonda com threshold 0.5
    sel = (x_frac >= 0.5).astype(int)

    # 2) Se ficar inviável, remover itens de pior razão até viabilizar
    def razoes():
        return valores / np.maximum(pesos, 1e-12)

    valor, peso = avaliar(sel, valores, pesos)
    if peso > capacidade:
        idx_selecionados = np.where(sel == 1)[0]
        ordem_remocao = idx_selecionados[np.argsort(razoes()[idx_selecionados])]  # pior razão primeiro
        for i in ordem_remocao:
            if peso <= capacidade:
                break
            sel[i] = 0
            valor -= valores[i]
            peso -= pesos[i]

    # 3) Se sobrar capacidade, tentar incluir itens de melhor razão
    if peso < capacidade:
        idx_nao_sel = np.where(sel == 0)[0]
        ordem_add = idx_nao_sel[np.argsort(-(razoes()[idx_nao_sel]))]
        for j in ordem_add:
            if peso + pesos[j] <= capacidade:
                sel[j] = 1
                valor += valores[j]
                peso += pesos[j]

    # 4) Competidor: melhor item único que cabe
    single_sel, single_val, single_peso = melhor_item_que_cabe(valores, pesos, capacidade)

    # 5) Competidor: solução gulosa
    if selecao_gulosa is None:
        selecao_gulosa = gulosa_ratio(valores, pesos, capacidade).selecao
    g_val, g_peso = avaliar(selecao_gulosa, valores, pesos)

    # 6) Escolhe o melhor entre os candidatos
    candidatos = [
        (sel, valor, peso),
        (single_sel, single_val, single_peso),
        (selecao_gulosa, g_val, g_peso),
    ]
    best_idx = int(np.argmax([c[1] for c in candidatos]))
    best_sel, best_val, best_peso = candidatos[best_idx]

    tempo_ms = (time.perf_counter() - inicio) * 1000.0
    return Solucao(best_sel.astype(int), float(best_val), float(best_peso), tempo_ms, metodo="Arredondamento")


# ============================
# Busca Local (1-add e 1-1 swap)
# ============================

def busca_local(
    valores: np.ndarray,
    pesos: np.ndarray,
    capacidade: float,
    selecao_inicial: np.ndarray | None = None,
    max_iter: int = 10000,
) -> Solucao:
    inicio = time.perf_counter()
    n = len(valores)
    if selecao_inicial is None:
        selecao = gulosa_ratio(valores, pesos, capacidade).selecao.copy()
    else:
        selecao = selecao_inicial.astype(int).copy()

    valor, peso = avaliar(selecao, valores, pesos)
    iteracoes = 0

    while iteracoes < max_iter:
        iteracoes += 1
        melhorou = False

        # 1) Tentativa de adição de um único item
        indices_nao = np.where(selecao == 0)[0]
        for j in indices_nao:
            if peso + pesos[j] <= capacidade and valores[j] > 0:
                novo_valor = valor + valores[j]
                if novo_valor > valor:
                    selecao[j] = 1
                    valor = novo_valor
                    peso += pesos[j]
                    melhorou = True
                    break
        if melhorou:
            continue

        # 2) Tentativa de troca 1-por-1 (swap)
        indices_sim = np.where(selecao == 1)[0]
        for i in indices_sim:
            for j in indices_nao:
                delta_peso = pesos[j] - pesos[i]
                if peso + delta_peso <= capacidade:
                    delta_valor = valores[j] - valores[i]
                    if delta_valor > 0:  # primeira melhoria
                        selecao[i] = 0
                        selecao[j] = 1
                        peso += delta_peso
                        valor += delta_valor
                        melhorou = True
                        break
            if melhorou:
                break

        if not melhorou:
            break  # ótimo local

    tempo_ms = (time.perf_counter() - inicio) * 1000.0
    return Solucao(selecao, float(valor), float(peso), tempo_ms, metodo="Busca Local")


# ============================
# Relatórios
# ============================

def detalhar_solucao(instancia: str, items: List[str], sol: Solucao) -> str:
    nomes = [items[i] for i in np.where(sol.selecao == 1)[0]]
    return (
        f"\nInstância: {os.path.basename(instancia)}\n"
        f"Método: {sol.metodo}\n"
        f"Itens selecionados ({len(nomes)}): {', '.join(nomes) if nomes else '(nenhum)'}\n"
        f"Valor total: {sol.valor:.2f} | Peso total: {sol.peso:.2f} | Tempo: {sol.tempo_ms:.2f} ms\n"
    )


def executar_instancia(csv_path: str) -> Dict[str, object]:
    items, valores, pesos, capacidade = carregar_instancia_mochila(csv_path)

    # Gulosa
    sol_g = gulosa_ratio(valores, pesos, capacidade)

    # Relaxação Linear (Upper Bound)
    x_frac, ub, peso_frac, t_rl = relaxacao_linear(valores, pesos, capacidade)

    # Arredondamento (usa gulosa como candidato também)
    sol_a = arredondamento_reparo(x_frac, valores, pesos, capacidade, selecao_gulosa=sol_g.selecao)

    # Busca Local (parte da melhor entre gulosa e arredondamento)
    base_sel = sol_g.selecao if sol_g.valor >= sol_a.valor else sol_a.selecao
    sol_bl = busca_local(valores, pesos, capacidade, selecao_inicial=base_sel)

    # Melhor solução entre as heurísticas puramente inteiras
    melhor_sol = max([sol_g, sol_a, sol_bl], key=lambda s: s.valor)

    # Gaps em relação ao upper bound da relaxação
    eps = 1e-12
    gap_g = 100.0 * max(0.0, (ub - sol_g.valor)) / max(ub, eps)
    gap_a = 100.0 * max(0.0, (ub - sol_a.valor)) / max(ub, eps)
    gap_bl = 100.0 * max(0.0, (ub - sol_bl.valor)) / max(ub, eps)

    # Impressões detalhadas
    print(detalhar_solucao(csv_path, items, sol_g))
    print(f"Relaxação Linear: UB={ub:.2f} | Peso frac.: {peso_frac:.2f} | Tempo: {t_rl:.2f} ms\n")
    print(detalhar_solucao(csv_path, items, sol_a))
    print(detalhar_solucao(csv_path, items, sol_bl))
    print(f"Melhor Heurística: {melhor_sol.metodo} (Valor={melhor_sol.valor:.2f}, Peso={melhor_sol.peso:.2f})\n")

    return {
        "Instancia": os.path.basename(csv_path),
        "n_itens": len(items),
        "Capacidade": capacidade,
        "Greedy_valor": sol_g.valor,
        "Greedy_peso": sol_g.peso,
        "Greedy_tempo_ms": sol_g.tempo_ms,
        "RL_UB": ub,
        "RL_tempo_ms": t_rl,
        "Arred_valor": sol_a.valor,
        "Arred_peso": sol_a.peso,
        "Arred_tempo_ms": sol_a.tempo_ms,
        "BL_valor": sol_bl.valor,
        "BL_peso": sol_bl.peso,
        "BL_tempo_ms": sol_bl.tempo_ms,
        "Gap_Greedy_%": gap_g,
        "Gap_Arred_%": gap_a,
        "Gap_BL_%": gap_bl,
        "Melhor_metodo": melhor_sol.metodo,
        "Melhor_valor": melhor_sol.valor,
    }


def executar_comparacao(instancias: List[str]) -> pd.DataFrame:
    resultados = []
    print("=" * 90)
    print("ANÁLISE COMPARATIVA — Heurísticas para a Mochila 0-1")
    print("=" * 90)
    for path in instancias:
        try:
            resultados.append(executar_instancia(path))
            print("-" * 90)
        except Exception as e:
            print(f"[ERRO] Falha ao processar {path}: {e}")
            print("-" * 90)
    df = pd.DataFrame(resultados)
    if not df.empty:
        # Ordena por quantidade de itens (se existir)
        df = df.sort_values(by=["n_itens", "Instancia"], ascending=[True, True])
    return df


def imprimir_tabelas(df: pd.DataFrame) -> None:
    if df.empty:
        print("Nenhum resultado para exibir.")
        return
    cols_visao_geral = [
        "Instancia", "n_itens", "Capacidade",
        "Greedy_valor", "Arred_valor", "BL_valor", "RL_UB",
        "Gap_Greedy_%", "Gap_Arred_%", "Gap_BL_%",
        "Melhor_metodo", "Melhor_valor",
    ]
    print("\nResumo Comparativo (valores e gaps):")
    print(df[cols_visao_geral].to_string(index=False, justify="center"))

    cols_tempo = [
        "Instancia", "Greedy_tempo_ms", "Arred_tempo_ms", "BL_tempo_ms", "RL_tempo_ms"
    ]
    print("\nTempos (ms):")
    print(df[cols_tempo].to_string(index=False, justify="center"))


if __name__ == "__main__":
    # Lista padrão de instâncias fornecidas em data/
    instancias = [
        os.path.join("data", "C1.csv"),
        os.path.join("data", "C2.csv"),
        os.path.join("data", "C3.csv"),
        os.path.join("data", "C4.csv"),
    ]

    # Opcional: incluir itens_mochila.csv se desejar adicionar mais um cenário
    extra = os.path.join("data", "itens_mochila.csv")
    if os.path.exists(extra):
        instancias.append(extra)

    df_result = executar_comparacao(instancias)
    imprimir_tabelas(df_result)
