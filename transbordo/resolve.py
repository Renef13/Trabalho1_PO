from pyomo.environ import SolverFactory
from modelo import criar_modelo
from instancias import INSTANCIAS


def resolver_instancia(nome_instancia, nome_solver="glpk"):
    """
    Resolve uma instância do problema

    Args:
        nome_instancia (str): "A", "B" ou "C"
        nome_solver (str): Nome do solver (glpk, gurobi, cplex, ipopt)

    Returns:
        model: Modelo resolvido
        results: Resultados da otimização
    """

    print("=" * 70)
    print(f"RESOLVENDO INSTÂNCIA {nome_instancia}")
    print("=" * 70)

    # 1. Obter dados da instância
    if nome_instancia not in INSTANCIAS:
        raise ValueError(f"Instância '{nome_instancia}' não encontrada. Use 'A', 'B' ou 'C'.")

    dados = INSTANCIAS[nome_instancia]

    # 2. Criar modelo
    print("\n[1/3] Criando modelo...")
    model = criar_modelo(dados)
    print(f"      ✓ Variáveis: {model.nvariables()}")
    print(f"      ✓ Restrições: {model.nconstraints()}")

    # 3. Declarar solver explicitamente
    print(f"\n[2/3] Configurando solver: {nome_solver.upper()}")
    solver = SolverFactory(nome_solver)

    # Configurações opcionais do solver
    if nome_solver == "glpk":
        solver.options['tmlim'] = 300  # Limite de tempo: 5 minutos
    elif nome_solver == "gurobi":
        solver.options['TimeLimit'] = 300
        solver.options['MIPGap'] = 0.01
    elif nome_solver == "cplex":
        solver.options['timelimit'] = 300
        solver.options['mipgap'] = 0.01

    # 4. Resolver
    print(f"\n[3/3] Resolvendo modelo...")
    results = solver.solve(model, tee=True)

    # 5. Verificar status da solução
    print("\n" + "=" * 70)
    print("STATUS DA SOLUÇÃO")
    print("=" * 70)

    if results.solver.termination_condition == "optimal":
        print("✓ Solução ótima encontrada!")
    elif results.solver.termination_condition == "feasible":
        print("⚠ Solução viável encontrada (pode não ser ótima)")
    else:
        print(f"✗ Erro: {results.solver.termination_condition}")
        return model, results

    return model, results


def exibir_resultados(model):
    """
    Exibe os resultados da otimização de forma organizada

    Args:
        model: Modelo Pyomo resolvido
    """

    print("\n" + "=" * 70)
    print("RESULTADOS DA OTIMIZAÇÃO")
    print("=" * 70)

    # Custo total
    print(f"\n💰 CUSTO TOTAL: ${model.objetivo():.2f}")

    # Decomposição dos custos
    custo_silos = sum(model.cf_silo[j] * model.z[j]() for j in model.S)
    custo_carrocerias = sum(model.cf_carr * model.t[i, j]() for (i, j) in model.Arcos_FS)
    custo_transporte = sum(model.custo_ferro[j, k] * model.y[j, k]() for (j, k) in model.Arcos_SP)

    print(f"\n   • Custo fixo dos silos:        ${custo_silos:.2f}")
    print(f"   • Custo fixo das carrocerias:  ${custo_carrocerias:.2f}")
    print(f"   • Custo de transporte (ferro): ${custo_transporte:.2f}")

    # Silos ativados
    print("\n" + "-" * 70)
    print("🏭 SILOS ATIVADOS")
    print("-" * 70)

    silos_ativos = [j for j in model.S if model.z[j]() > 0.5]

    if silos_ativos:
        for j in silos_ativos:
            fluxo_entrada = sum(model.x[i, j]() for i in model.F)
            utilizacao = (fluxo_entrada / model.cap_silo[j]) * 100
            print(
                f"   {j}: {fluxo_entrada:.2f}t / {model.cap_silo[j]}t ({utilizacao:.1f}% utilizado) - Custo fixo: ${model.cf_silo[j]}")
    else:
        print("   Nenhum silo ativado")

    # Portos ativos
    print("\n" + "-" * 70)
    print("🚢 PORTOS ATIVOS")
    print("-" * 70)

    portos_ativos = [k for k in model.P if model.w[k]() > 0.5]

    if portos_ativos:
        for k in portos_ativos:
            fluxo_recebido = sum(model.y[j, k]() for j in model.S)
            atendimento = (fluxo_recebido / model.dem[k]) * 100
            print(f"   {k}: Recebeu {fluxo_recebido:.2f}t (Demanda: {model.dem[k]}t - {atendimento:.1f}% atendida)")
    else:
        print("   Nenhum porto ativado")

    # Fluxos Fazenda → Silo
    print("\n" + "-" * 70)
    print("🚚 FLUXOS FAZENDA → SILO (Treminhões)")
    print("-" * 70)

    fluxos_fs = [(i, j, model.x[i, j](), model.t[i, j]())
                 for (i, j) in model.Arcos_FS if model.x[i, j]() > 0.01]

    if fluxos_fs:
        for i, j, fluxo, carrocerias in fluxos_fs:
            print(f"   {i} → {j}: {fluxo:.2f}t usando {int(carrocerias)} carroceria(s)")
    else:
        print("   Nenhum fluxo")

    # Fluxos Silo → Porto
    print("\n" + "-" * 70)
    print("🚂 FLUXOS SILO → PORTO (Ferrovia)")
    print("-" * 70)

    fluxos_sp = [(j, k, model.y[j, k](), model.custo_ferro[j, k])
                 for (j, k) in model.Arcos_SP if model.y[j, k]() > 0.01]

    if fluxos_sp:
        for j, k, fluxo, custo_unitario in fluxos_sp:
            custo_trecho = fluxo * custo_unitario
            print(f"   {j} → {k}: {fluxo:.2f}t (Custo: ${custo_unitario}/t = ${custo_trecho:.2f} total)")
    else:
        print("   Nenhum fluxo")

    # Resumo de produção e demanda
    print("\n" + "-" * 70)
    print("📊 BALANÇO GERAL")
    print("-" * 70)

    prod_total = sum(model.prod[i] for i in model.F)
    dem_total = sum(model.dem[k] for k in model.P)
    enviado_total = sum(model.y[j, k]() for (j, k) in model.Arcos_SP)

    print(f"   Produção total das fazendas: {prod_total}t")
    print(f"   Demanda total dos portos:    {dem_total}t")
    print(f"   Total enviado aos portos:    {enviado_total:.2f}t")

    if enviado_total < dem_total:
        deficit = dem_total - enviado_total
        print(f"   ⚠ Déficit: {deficit:.2f}t (produção insuficiente)")


def main():
    """
    Função principal
    """

    # ===== CONFIGURAÇÃO =====
    # Escolha a instância: "A", "B" ou "C"
    INSTANCIA = "C"

    # Escolha o solver: "glpk", "gurobi", "cplex", "ipopt"
    SOLVER = "glpk"

    # ===== RESOLUÇÃO =====
    try:

        model, results = resolver_instancia(INSTANCIA, SOLVER)

        if results.solver.termination_condition in ["optimal", "feasible"]:
            exibir_resultados(model)

    except Exception as e:
        print(f"\n ERRO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()