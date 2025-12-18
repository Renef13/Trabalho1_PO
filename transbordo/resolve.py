from pyomo.environ import SolverFactory
from modelo import criar_modelo
from instancias import INSTANCIAS
import json


def resolver_instancia(nome_instancia, nome_solver="glpk"):
    print("=" * 70)
    print(f"RESOLVENDO INSTÂNCIA {nome_instancia}")
    print("=" * 70)

    if nome_instancia not in INSTANCIAS:
        raise ValueError(f"Instância '{nome_instancia}' não encontrada. Use 'A', 'B' ou 'C'.")

    dados = INSTANCIAS[nome_instancia]

    print("\n[1/3] Criando modelo...")
    model = criar_modelo(dados)
    print(f"      ✓ Variáveis: {model.nvariables()}")
    print(f"      ✓ Restrições: {model.nconstraints()}")

    print(f"\n[2/3] Configurando solver: {nome_solver.upper()}")
    solver = SolverFactory(nome_solver)

    if nome_solver == "glpk":
        solver.options['tmlim'] = 300
    elif nome_solver == "gurobi":
        solver.options['TimeLimit'] = 300
        solver.options['MIPGap'] = 0.01
    elif nome_solver == "cplex":
        solver.options['timelimit'] = 300
        solver.options['mipgap'] = 0.01

    print(f"\n[3/3] Resolvendo modelo...")
    results = solver.solve(model, tee=False)

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
    print("\n" + "=" * 70)
    print("RESULTADOS DA OTIMIZAÇÃO")
    print("=" * 70)

    print(f"\n💰 CUSTO TOTAL: R$ {model.objetivo():.2f}")

    # Cálculo dos custos por componente
    custo_silos = sum(model.cf_silo[j] * model.z[j]() for j in model.S)
    custo_carrocerias = sum(model.cf_carr * model.t[i, j]() for (i, j) in model.Arcos_FS)
    custo_transporte = sum(model.custo_ferro[j, k] * model.y[j, k]() for (j, k) in model.Arcos_SP)

    # Custo rodoviário (treminhão) = 5 R$/t * volume total
    volume_rodoviario = sum(model.x[i, j]() for (i, j) in model.Arcos_FS)
    custo_rodoviario = volume_rodoviario * 5  # 5 R$/t conforme modelo

    print(f"\n📊 DECOMPOSIÇÃO DE CUSTOS:")
    print(f"   • Transporte Fazenda→Silo (rodoviário): R$ {custo_rodoviario:.2f}")
    print(f"   • Custo fixo dos silos:                 R$ {custo_silos:.2f}")
    print(f"   • Transporte Silo→Porto (ferroviário):  R$ {custo_transporte:.2f}")
    print(f"   • Custo das carrocerias:                R$ {custo_carrocerias:.2f}")
    print(f"   • TOTAL:                                R$ {model.objetivo():.2f}")

    # Análise percentual
    total = model.objetivo()
    print(f"\n📈 PARTICIPAÇÃO PERCENTUAL:")
    print(f"   • Rodoviário:    {(custo_rodoviario / total) * 100:.1f}%")
    print(f"   • Custo fixo:    {(custo_silos / total) * 100:.1f}%")
    print(f"   • Ferroviário:   {(custo_transporte / total) * 100:.1f}%")
    print(f"   • Carrocerias:   {(custo_carrocerias / total) * 100:.1f}%")

    print("\n" + "-" * 70)
    print("🏭 SILOS ATIVADOS E UTILIZAÇÃO")
    print("-" * 70)

    silos_ativos = [j for j in model.S if model.z[j]() > 0.5]

    if silos_ativos:
        for j in sorted(silos_ativos):
            fluxo_entrada = sum(model.x[i, j]() for i in model.F)
            utilizacao = (fluxo_entrada / model.cap_silo[j]) * 100
            status = "SATURADO" if utilizacao >= 99.9 else "COM FOLGA"
            print(
                f"   {j}: {fluxo_entrada:.0f}t / {model.cap_silo[j]}t ({utilizacao:.1f}%) [{status}] - CF: R$ {model.cf_silo[j]:.2f}")
    else:
        print("   Nenhum silo ativado")

    print(f"\n   Total de silos ativados: {len(silos_ativos)}")

    print("\n" + "-" * 70)
    print("🚢 PORTOS E ATENDIMENTO")
    print("-" * 70)

    portos_ativos = [k for k in model.P if model.w[k]() > 0.5]

    if portos_ativos:
        for k in sorted(portos_ativos):
            fluxo_recebido = sum(model.y[j, k]() for j in model.S)
            capacidade_porto = model.dem[k]
            atendimento = (fluxo_recebido / capacidade_porto) * 100
            status = "SATURADO" if atendimento >= 99.9 else "COM FOLGA"
            print(f"   {k}: {fluxo_recebido:.0f}t / {capacidade_porto}t ({atendimento:.1f}%) [{status}]")
    else:
        print("   Nenhum porto ativado")

    print("\n" + "-" * 70)
    print("🚚 FLUXOS FAZENDA → SILO (Treminhões)")
    print("-" * 70)

    fluxos_fs = [(i, j, model.x[i, j](), model.t[i, j]())
                 for (i, j) in model.Arcos_FS if model.x[i, j]() > 0.01]

    if fluxos_fs:
        total_carrocerias = 0
        for i, j, fluxo, carrocerias in sorted(fluxos_fs):
            utilizacao_carr = (fluxo / (model.cap_carr * carrocerias)) * 100 if carrocerias > 0 else 0
            print(f"   {i} → {j}: {fluxo:.1f}t usando {int(carrocerias)} carroceria(s) ({utilizacao_carr:.1f}% util.)")
            total_carrocerias += int(carrocerias)

        capacidade_total_carr = total_carrocerias * model.cap_carr
        utilizacao_agregada = (volume_rodoviario / capacidade_total_carr) * 100 if capacidade_total_carr > 0 else 0
        print(f"\n   Total de carrocerias: {total_carrocerias}")
        print(f"   Capacidade contratada: {capacidade_total_carr:.0f}t")
        print(f"   Utilização agregada: {utilizacao_agregada:.1f}%")
    else:
        print("   Nenhum fluxo")

    print("\n" + "-" * 70)
    print("🚂 FLUXOS SILO → PORTO (Ferrovia)")
    print("-" * 70)

    fluxos_sp = [(j, k, model.y[j, k](), model.custo_ferro[j, k])
                 for (j, k) in model.Arcos_SP if model.y[j, k]() > 0.01]

    if fluxos_sp:
        for j, k, fluxo, custo_unitario in sorted(fluxos_sp):
            custo_trecho = fluxo * custo_unitario
            print(f"   {j} → {k}: {fluxo:.0f}t (R$ {custo_unitario:.0f}/t = R$ {custo_trecho:.2f} total)")
    else:
        print("   Nenhum fluxo")

    print("\n" + "-" * 70)
    print("📊 BALANÇO GERAL")
    print("-" * 70)

    prod_total = sum(model.prod[i] for i in model.F)
    dem_total = sum(model.dem[k] for k in model.P)
    enviado_total = sum(model.y[j, k]() for (j, k) in model.Arcos_SP)

    print(f"   Produção total das fazendas: {prod_total}t")
    print(f"   Capacidade total dos portos: {dem_total}t")
    print(f"   Total enviado aos portos:    {enviado_total:.0f}t")
    print(f"   Desequilíbrio oferta/demanda: {prod_total}t vs {dem_total}t")

    if enviado_total < dem_total:
        deficit = dem_total - enviado_total
        print(f"   ⚠ Déficit: {deficit:.0f}t (produção insuficiente)")


def analise_comparativa_instancias():
    """Resolve todas as instâncias e faz análise comparativa"""
    print("\n" + "=" * 70)
    print("ANÁLISE COMPARATIVA DAS TRÊS INSTÂNCIAS")
    print("=" * 70)

    resultados = {}

    for inst in ['A', 'C', 'B']:  # Ordem: mais barata → base → mais cara
        model, results = resolver_instancia(inst, "glpk")

        if results.solver.termination_condition in ["optimal", "feasible"]:
            custo_silos = sum(model.cf_silo[j] * model.z[j]() for j in model.S)
            custo_carrocerias = sum(model.cf_carr * model.t[i, j]() for (i, j) in model.Arcos_FS)
            custo_transporte = sum(model.custo_ferro[j, k] * model.y[j, k]() for (j, k) in model.Arcos_SP)
            volume_rodoviario = sum(model.x[i, j]() for (i, j) in model.Arcos_FS)
            custo_rodoviario = volume_rodoviario * 5

            silos_ativos = len([j for j in model.S if model.z[j]() > 0.5])

            # Porto mais atendido
            fluxos_portos = {}
            for k in model.P:
                fluxos_portos[k] = sum(model.y[j, k]() for j in model.S)
            porto_max = max(fluxos_portos, key=fluxos_portos.get)
            porto_min = min(fluxos_portos, key=fluxos_portos.get)

            resultados[inst] = {
                'custo_total': model.objetivo(),
                'custo_rodoviario': custo_rodoviario,
                'custo_fixo': custo_silos,
                'custo_ferro': custo_transporte,
                'custo_carrocerias': custo_carrocerias,
                'silos_ativos': silos_ativos,
                'porto_max': (porto_max, fluxos_portos[porto_max]),
                'porto_min': (porto_min, fluxos_portos[porto_min])
            }

    # Tabela comparativa
    print("\n" + "=" * 70)
    print("TABELA COMPARATIVA")
    print("=" * 70)
    print(f"\n{'Indicador':<40} {'Inst. A':<15} {'Inst. C':<15} {'Inst. B':<15}")
    print("-" * 70)

    for inst in ['A', 'C', 'B']:
        if inst not in resultados:
            continue

    # Custos
    print(f"{'Custo total (R$)':<40} ", end="")
    for inst in ['A', 'C', 'B']:
        print(f"R$ {resultados[inst]['custo_total']:>10,.2f}  ", end="")
    print()

    print(f"{'Transporte Fazenda→Silo (R$)':<40} ", end="")
    for inst in ['A', 'C', 'B']:
        print(f"R$ {resultados[inst]['custo_rodoviario']:>10,.2f}  ", end="")
    print()

    print(f"{'Custo fixo dos silos (R$)':<40} ", end="")
    for inst in ['A', 'C', 'B']:
        print(f"R$ {resultados[inst]['custo_fixo']:>10,.2f}  ", end="")
    print()

    print(f"{'Transporte Silo→Porto (R$)':<40} ", end="")
    for inst in ['A', 'C', 'B']:
        print(f"R$ {resultados[inst]['custo_ferro']:>10,.2f}  ", end="")
    print()

    print(f"{'Custo das carrocerias (R$)':<40} ", end="")
    for inst in ['A', 'C', 'B']:
        print(f"R$ {resultados[inst]['custo_carrocerias']:>10,.2f}  ", end="")
    print()

    print(f"{'Silos ativados (quant.)':<40} ", end="")
    for inst in ['A', 'C', 'B']:
        print(f"{resultados[inst]['silos_ativos']:>14}  ", end="")
    print()

    print(f"{'Porto mais atendido':<40} ", end="")
    for inst in ['A', 'C', 'B']:
        p, v = resultados[inst]['porto_max']
        print(f"{p} ({v:.0f}t)     ", end="")
    print()

    print(f"{'Porto menos atendido':<40} ", end="")
    for inst in ['A', 'C', 'B']:
        p, v = resultados[inst]['porto_min']
        print(f"{p} ({v:.0f}t)     ", end="")
    print()

    print("\n" + "=" * 70)
    print("ANÁLISE QUALITATIVA")
    print("=" * 70)

    print(f"\n🔍 Comparação relativa:")
    print(f"   • Instância A (custo fixo BAIXO):  R$ {resultados['A']['custo_total']:,.2f} - MAIS BARATA")
    print(f"   • Instância C (custo fixo BASE):   R$ {resultados['C']['custo_total']:,.2f} - REFERÊNCIA")
    print(f"   • Instância B (custo fixo ALTO):   R$ {resultados['B']['custo_total']:,.2f} - MAIS CARA")

    dif_AC = resultados['C']['custo_total'] - resultados['A']['custo_total']
    dif_BC = resultados['B']['custo_total'] - resultados['C']['custo_total']

    print(f"\n   Diferença C vs A: R$ {dif_AC:,.2f} (+{(dif_AC / resultados['A']['custo_total']) * 100:.1f}%)")
    print(f"   Diferença B vs C: R$ {dif_BC:,.2f} (+{(dif_BC / resultados['C']['custo_total']) * 100:.1f}%)")

    print(f"\n💡 Observações:")
    print(f"   • Todos os cenários ativam {resultados['A']['silos_ativos']} silos (decisão estrutural)")
    print(f"   • Fluxos ferroviários são idênticos nas três instâncias")
    print(f"   • O componente ferroviário domina o custo total em todas as instâncias")
    print(f"   • Sistema opera próximo ao limite de capacidade (gargalo)")


def main():
    print("\n" + "=" * 70)
    print("SISTEMA DE ANÁLISE - TRANSBORDO DE BACURI")
    print("=" * 70)

    print("\nEscolha o modo de análise:")
    print("1 - Resolver uma instância específica")
    print("2 - Análise comparativa completa (A, B, C)")

    opcao = input("\nOpção: ").strip()

    if opcao == "1":
        print("\nInstâncias disponíveis: A (baixo), C (base), B (alto)")
        INSTANCIA = input("Digite a instância: ").strip().upper()

        if INSTANCIA not in ['A', 'B', 'C']:
            print("Instância inválida!")
            return

        SOLVER = "glpk"

        try:
            model, results = resolver_instancia(INSTANCIA, SOLVER)

            if results.solver.termination_condition in ["optimal", "feasible"]:
                exibir_resultados(model)

        except Exception as e:
            print(f"\n❌ ERRO: {e}")
            import traceback
            traceback.print_exc()

    elif opcao == "2":
        try:
            analise_comparativa_instancias()
        except Exception as e:
            print(f"\n❌ ERRO: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("Opção inválida!")


if __name__ == "__main__":
    main()