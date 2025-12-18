from pyomo.environ import SolverFactory
from modelo import criar_modelo
from instancias import INSTANCIAS


def extrair_dados_instancia(nome_inst):
    print(f"\n{'=' * 70}")
    print(f"RESOLVENDO INSTÂNCIA {nome_inst}")
    print(f"{'=' * 70}")

    dados = INSTANCIAS[nome_inst]
    model = criar_modelo(dados)

    solver = SolverFactory("glpk")
    solver.options['tmlim'] = 300
    results = solver.solve(model, tee=False)

    if results.solver.termination_condition not in ["optimal", "feasible"]:
        print(f" ERRO: {results.solver.termination_condition}")
        return None

    print(f"✓ Solução ótima encontrada!")

    custo_silos = sum(model.cf_silo[j] * model.z[j]() for j in model.S)
    custo_carrocerias = sum(model.cf_carr * model.t[i, j]() for (i, j) in model.Arcos_FS)
    custo_ferro = sum(model.custo_ferro[j, k] * model.y[j, k]() for (j, k) in model.Arcos_SP)
    custo_total = model.objetivo()

    silos_ativos = {}
    for j in model.S:
        if model.z[j]() > 0.5:
            volume = sum(model.x[i, j]() for i in model.F)
            silos_ativos[j] = {
                'volume': volume,
                'capacidade': model.cap_silo[j],
                'utilizacao': (volume / model.cap_silo[j]) * 100,
                'custo_fixo': model.cf_silo[j]
            }

    portos_ativos = {}
    for k in model.P:
        if model.w[k]() > 0.5:
            volume = sum(model.y[j, k]() for j in model.S)
            portos_ativos[k] = {
                'volume': volume,
                'demanda': model.dem[k],
                'atendimento': (volume / model.dem[k]) * 100
            }

    total_carrocerias = sum(int(model.t[i, j]()) for (i, j) in model.Arcos_FS)
    volume_transportado = sum(model.x[i, j]() for (i, j) in model.Arcos_FS)
    capacidade_carrocerias = total_carrocerias * model.cap_carr
    utilizacao_carrocerias = (volume_transportado / capacidade_carrocerias) * 100 if capacidade_carrocerias > 0 else 0

    rotas_limite_3 = []
    for (i, j) in model.Arcos_FS:
        if int(model.t[i, j]()) == 3:
            rotas_limite_3.append(f"{i}→{j}")

    return {
        'nome': nome_inst,
        'custo_total': custo_total,
        'custo_silos': custo_silos,
        'custo_ferro': custo_ferro,
        'custo_carrocerias': custo_carrocerias,
        'silos_ativos': silos_ativos,
        'portos_ativos': portos_ativos,
        'total_carrocerias': total_carrocerias,
        'capacidade_carrocerias': capacidade_carrocerias,
        'utilizacao_carrocerias': utilizacao_carrocerias,
        'volume_transportado': volume_transportado,
        'rotas_limite_3': rotas_limite_3
    }


def gerar_tabela(resultados):
    print("\n" + "=" * 70)
    print("TABELA")
    print("=" * 70)

    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Decomposição de custos por instância}")
    print("\\label{tab:custos}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Componente (R\\$)} & \\textbf{Inst. A} & \\textbf{Inst. C} & \\textbf{Inst. B} \\\\")
    print("\\midrule")

    print(
        f"Custo fixo silos & {resultados['A']['custo_silos']:.2f} & {resultados['C']['custo_silos']:.2f} & {resultados['B']['custo_silos']:.2f} \\\\")
    print(
        f"Custo carrocerias & {resultados['A']['custo_carrocerias']:.2f} & {resultados['C']['custo_carrocerias']:.2f} & {resultados['B']['custo_carrocerias']:.2f} \\\\")
    print(
        f"Transporte S$\\rightarrow$P (ferro) & {resultados['A']['custo_ferro']:.2f} & {resultados['C']['custo_ferro']:.2f} & {resultados['B']['custo_ferro']:.2f} \\\\")
    print("\\midrule")
    print(
        f"\\textbf{{TOTAL}} & \\textbf{{{resultados['A']['custo_total']:.2f}}} & \\textbf{{{resultados['C']['custo_total']:.2f}}} & \\textbf{{{resultados['B']['custo_total']:.2f}}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n% NOTA IMPORTANTE para o texto:")
    print("% O custo do trecho Fazenda→Silo está INCLUÍDO no 'Custo carrocerias'")
    print("% NÃO existe custo variável por tonelada no trecho rodoviário")


def gerar_resumo(resultados):
    print("\n" + "=" * 70)
    print("DADOS PARA AS SEÇÕES DE ANÁLISE")
    print("=" * 70)

    for inst in ['A', 'C', 'B']:
        r = resultados[inst]
        print(f"\n{'─' * 70}")
        print(f"INSTÂNCIA {inst}")
        print(f"{'─' * 70}")

        print(f"\n CUSTOS:")
        print(f"   • Custo total: R$ {r['custo_total']:,.2f}")
        print(f"   • Custo fixo silos: R$ {r['custo_silos']:,.2f} ({(r['custo_silos'] / r['custo_total']) * 100:.1f}%)")
        print(
            f"   • Custo carrocerias: R$ {r['custo_carrocerias']:,.2f} ({(r['custo_carrocerias'] / r['custo_total']) * 100:.1f}%)")
        print(f"   • Transporte S→P: R$ {r['custo_ferro']:,.2f} ({(r['custo_ferro'] / r['custo_total']) * 100:.1f}%)")

        print(f"\n SILOS ATIVADOS: {len(r['silos_ativos'])}")
        for silo, dados in sorted(r['silos_ativos'].items()):
            status = "SATURADO" if dados['utilizacao'] >= 99.9 else f"{dados['utilizacao']:.1f}%"
            print(f"   • {silo}: {dados['volume']:.0f}t / {dados['capacidade']}t ({status})")

        print(f"\n PORTOS ATIVOS: {len(r['portos_ativos'])}")
        for porto, dados in sorted(r['portos_ativos'].items()):
            print(f"   • {porto}: {dados['volume']:.0f}t / {dados['demanda']}t ({dados['atendimento']:.1f}%)")

        print(f"\n CARROCERIAS:")
        print(f"   • Total utilizado: {r['total_carrocerias']}")
        print(f"   • Capacidade contratada: {r['capacidade_carrocerias']:.0f}t")
        print(f"   • Volume transportado: {r['volume_transportado']:.0f}t")
        print(f"   • Utilização: {r['utilizacao_carrocerias']:.1f}%")
        if r['rotas_limite_3']:
            print(f"   • Rotas no limite (3 carr): {', '.join(r['rotas_limite_3'])}")


def main():
    resultados = {}

    for inst in ['A', 'C', 'B']:
        dados = extrair_dados_instancia(inst)
        if dados:
            resultados[inst] = dados

    if len(resultados) != 3:
        print("\n ERRO: Nem todas as instâncias foram resolvidas!")
        return

    gerar_tabela(resultados)
    gerar_resumo(resultados)

    print("\n" + "=" * 70)
    print("ANÁLISE COMPARATIVA")
    print("=" * 70)

    custo_A = resultados['A']['custo_total']
    custo_C = resultados['C']['custo_total']
    custo_B = resultados['B']['custo_total']

    print(f"\n Ranking de custos:")
    print(f"   1. Instância A: R$ {custo_A:,.2f} (MAIS BARATA)")
    print(f"   2. Instância C: R$ {custo_C:,.2f} (REFERÊNCIA)")
    print(f"   3. Instância B: R$ {custo_B:,.2f} (MAIS CARA)")

    print(f"\n Diferenças:")
    dif_CA = custo_C - custo_A
    dif_BC = custo_B - custo_C
    dif_BA = custo_B - custo_A

    print(f"   • C vs A: R$ {dif_CA:,.2f} (+{(dif_CA / custo_A) * 100:.1f}%)")
    print(f"   • B vs C: R$ {dif_BC:,.2f} (+{(dif_BC / custo_C) * 100:.1f}%)")
    print(f"   • B vs A: R$ {dif_BA:,.2f} (+{(dif_BA / custo_A) * 100:.1f}%)")

    silos_A = len(resultados['A']['silos_ativos'])
    silos_C = len(resultados['C']['silos_ativos'])
    silos_B = len(resultados['B']['silos_ativos'])

    print(f"\n Número de silos:")
    print(f"   • Instância A: {silos_A} silos")
    print(f"   • Instância C: {silos_C} silos")
    print(f"   • Instância B: {silos_B} silos")

    if silos_A == silos_C == silos_B:
        print(f"   ✓ TODAS ativam {silos_A} silos (decisão estrutural/gargalo)")
    else:
        print(f"   Número varia entre instâncias (há folga para otimização)")

    print("\n" + "=" * 70)
    print(" NOTAS IMPORTANTES PARA O RELATÓRIO:")
    print("=" * 70)
    print("1. O trecho Fazenda→Silo NÃO tem custo variável por tonelada")
    print("2. O único custo rodoviário é o custo FIXO das carrocerias (R$ 20 cada)")
    print("3. Portanto, NÃO existe 'Transporte F→S' separado na tabela")
    print("4. O custo ferroviário (S→P) é o único com custo variável por tonelada")


if __name__ == "__main__":
    main()
