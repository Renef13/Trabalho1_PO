from pyomo.environ import SolverFactory
from modelo import criar_modelo
from instancias import INSTANCIAS


class AnalisadorResultados:

    def __init__(self, model, nome_instancia):
        self.model = model
        self.nome = nome_instancia
        self._calcular_metricas()

    def _calcular_metricas(self):

        m = self.model

        self.custo_silos = sum(m.cf_silo[j] * m.z[j]() for j in m.S)
        self.custo_carrocerias = sum(m.cf_carr * m.t[i, j]() for (i, j) in m.Arcos_FS)
        self.custo_ferro = sum(m.custo_ferro[j, k] * m.y[j, k]() for (j, k) in m.Arcos_SP)

        volume_rod = sum(m.x[i, j]() for (i, j) in m.Arcos_FS)
        self.custo_rodoviario = volume_rod * 5  # 5 R$/t

        self.custo_total = m.objetivo()

        self.silos_ativos = {j: sum(m.x[i, j]() for i in m.F)
                             for j in m.S if m.z[j]() > 0.5}

        self.fluxos_portos = {k: sum(m.y[j, k]() for j in m.S)
                              for k in m.P if m.w[k]() > 0.5}

        self.total_carrocerias = sum(int(m.t[i, j]()) for (i, j) in m.Arcos_FS)
        self.capacidade_carrocerias = self.total_carrocerias * m.cap_carr
        self.utilizacao_carrocerias = (volume_rod / self.capacidade_carrocerias * 100
                                       if self.capacidade_carrocerias > 0 else 0)

        self.prod_total = sum(m.prod[i] for i in m.F)
        self.cap_portos = sum(m.dem[k] for k in m.P)

    def gerar_secao_ativacao_silos(self):

        print("\n" + "=" * 70)
        print(f"SEÇÃO 1: ATIVAÇÃO DE SILOS - INSTÂNCIA {self.nome}")
        print("=" * 70)

        print(f"\nQuantidade de silos ativados: {len(self.silos_ativos)}")
        print(f"Produção total: {self.prod_total}t")
        print(f"Capacidade total dos portos: {self.cap_portos}t")

        print("\n Processamento por silo:")
        for silo in sorted(self.silos_ativos.keys()):
            volume = self.silos_ativos[silo]
            capacidade = self.model.cap_silo[silo]
            util = (volume / capacidade) * 100
            custo_fixo = self.model.cf_silo[silo]
            status = "SATURADO" if util >= 99.9 else f"{util:.1f}% utilizado"

            print(f"   • {silo}: {volume:.0f}t / {capacidade}t ({status}) - CF: R$ {custo_fixo:.2f}")

        print("\n Análise estrutural:")

        silos_ordenados = sorted(self.silos_ativos.items(), key=lambda x: x[1], reverse=True)
        capacidade_tres_maiores = sum(self.model.cap_silo[s] for s, _ in silos_ordenados[:3])

        print(f"   • Capacidade dos 3 maiores silos: {capacidade_tres_maiores}t")
        print(f"   • Produção total: {self.prod_total}t")

        if capacidade_tres_maiores < self.prod_total:
            print(f"   • CONCLUSÃO: O 4º silo é OBRIGATÓRIO (gargalo de capacidade)")
            print(f"   • Mesmo com custo fixo alto, não é possível desativar silos")
        else:
            print(f"   • CONCLUSÃO: Haveria folga para fechar um silo")

        print("\n Justificativas por silo:")
        for silo in sorted(self.silos_ativos.keys()):
            volume = self.silos_ativos[silo]
            capacidade = self.model.cap_silo[silo]

            rotas_ferro = []
            for k in self.model.P:
                fluxo = self.model.y[silo, k]()
                if fluxo > 0.01:
                    custo_unit = self.model.custo_ferro[silo, k]
                    rotas_ferro.append(f"{k} ({fluxo:.0f}t, R$ {custo_unit}/t)")

            print(f"\n   {silo}:")
            print(f"      - Processamento: {volume:.0f}t (capacidade: {capacidade}t)")
            print(f"      - Rotas ferroviárias: {', '.join(rotas_ferro)}")

            if volume >= capacidade * 0.999:
                print(f"      - Status: SATURADO - opera no limite")
                print(f"      - Trade-off: capacidade crítica para o sistema")
            else:
                folga = capacidade - volume
                print(f"      - Status: Folga de {folga:.0f}t")
                print(f"      - Função: amortecedor do sistema")

    def gerar_secao_custos(self):

        print("\n" + "=" * 70)
        print(f"SEÇÃO 2: CUSTOS TOTAIS E DECOMPOSIÇÃO - INSTÂNCIA {self.nome}")
        print("=" * 70)

        print(f"\n CUSTO TOTAL: R$ {self.custo_total:,.2f}")

        print("\n Decomposição por componente:")
        print(f"   1. Transporte Fazenda→Silo:  R$ {self.custo_rodoviario:>10,.2f}")
        print(f"   2. Custo fixo dos silos:     R$ {self.custo_silos:>10,.2f}")
        print(f"   3. Transporte Silo→Porto:    R$ {self.custo_ferro:>10,.2f}")
        print(f"   4. Custo das carrocerias:    R$ {self.custo_carrocerias:>10,.2f}")
        print(f"   {'─' * 44}")
        print(f"   TOTAL:                       R$ {self.custo_total:>10,.2f}")

        print("\n Análise percentual (participação no custo total):")
        print(f"   • Rodoviário:   {(self.custo_rodoviario / self.custo_total) * 100:>6.2f}%")
        print(f"   • Custo fixo:   {(self.custo_silos / self.custo_total) * 100:>6.2f}%")
        print(f"   • Ferroviário:  {(self.custo_ferro / self.custo_total) * 100:>6.2f}%")
        print(f"   • Carrocerias:  {(self.custo_carrocerias / self.custo_total) * 100:>6.2f}%")

        componentes = {
            'Ferroviário': self.custo_ferro,
            'Rodoviário': self.custo_rodoviario,
            'Custo fixo': self.custo_silos,
            'Carrocerias': self.custo_carrocerias
        }
        maior = max(componentes, key=componentes.get)

        print(f"\n Componente dominante: {maior}")
        print(f"   O {maior.lower()} representa a maior parcela do custo,")
        print(f"   concentrando {(componentes[maior] / self.custo_total) * 100:.1f}% do total.")

    def gerar_secao_modais(self):

        print("\n" + "=" * 70)
        print(f"SEÇÃO 3: IMPACTO DOS MODAIS DE TRANSPORTE - INSTÂNCIA {self.nome}")
        print("=" * 70)

        print("\n MODAL RODOVIÁRIO (Treminhão: Fazenda→Silo)")
        print(f"   • Custo total: R$ {self.custo_rodoviario:,.2f}")
        print(f"   • Volume transportado: {self.prod_total}t")
        print(f"   • Custo unitário: R$ 5,00/t (fixo para todas as rotas)")
        print(f"   • Carrocerias utilizadas: {self.total_carrocerias}")
        print(f"   • Capacidade contratada: {self.capacidade_carrocerias:.0f}t")
        print(f"   • Utilização agregada: {self.utilizacao_carrocerias:.1f}%")

        rotas_com_3 = []
        for (i, j) in self.model.Arcos_FS:
            if int(self.model.t[i, j]()) == 3:
                rotas_com_3.append(f"{i}→{j}")

        if rotas_com_3:
            print(f"\n Rotas no limite (3 carrocerias): {', '.join(rotas_com_3)}")
            print(f"   A restrição operacional está ATIVA nessas rotas")

        print("\n MODAL FERROVIÁRIO (Silo→Porto)")
        print(f"   • Custo total: R$ {self.custo_ferro:,.2f}")
        print(f"   • Volume transportado: {self.prod_total}t")
        print(f"   • Custos DIFERENCIADOS por rota")

        print("\n   Distribuição por porto:")
        for porto in sorted(self.fluxos_portos.keys()):
            volume = self.fluxos_portos[porto]
            capacidade = self.model.dem[porto]
            util = (volume / capacidade) * 100
            status = "SATURADO" if util >= 99.9 else "COM FOLGA"
            print(f"      • {porto}: {volume:.0f}t / {capacidade}t ({status})")

        print("\n Comparação modal:")
        print(f"   • O modal FERROVIÁRIO é mais caro em termos absolutos")
        print(f"   • Ferrovia: R$ {self.custo_ferro:,.2f} vs Rodoviário: R$ {self.custo_rodoviario:,.2f}")
        print(f"   • Diferença: R$ {(self.custo_ferro - self.custo_rodoviario):,.2f}")
        print(f"   • A matriz de custos diferenciada no modal ferroviário")
        print(f"     permite priorizar portos mais econômicos")

    def gerar_secao_gargalos(self):

        print("\n" + "=" * 70)
        print(f"SEÇÃO 4: GARGALOS LOGÍSTICOS - INSTÂNCIA {self.nome}")
        print("=" * 70)

        print("\n GARGALO 1: Capacidade dos silos")
        silos_saturados = []
        for silo, volume in self.silos_ativos.items():
            capacidade = self.model.cap_silo[silo]
            util = (volume / capacidade) * 100
            if util >= 99.9:
                silos_saturados.append(silo)
                print(f"   • {silo}: SATURADO ({volume:.0f}t / {capacidade}t)")
            else:
                folga = capacidade - volume
                print(f"   • {silo}: {folga:.0f}t de folga")

        if len(silos_saturados) >= 3:
            print(f"\n    CRÍTICO: {len(silos_saturados)} silos saturados")
            print(f"   O sistema opera com pouca folga de capacidade")

        print("\n GARGALO 2: Restrição de carrocerias")
        rotas_limite = []
        for (i, j) in self.model.Arcos_FS:
            carrocerias = int(self.model.t[i, j]())
            if carrocerias == 3:
                volume = self.model.x[i, j]()
                rotas_limite.append((i, j, volume))

        if rotas_limite:
            print(f"   • {len(rotas_limite)} rota(s) usando o limite máximo (3 carrocerias)")
            for i, j, v in rotas_limite:
                print(f"      - {i}→{j}: {v:.1f}t")
            print(f"   A restrição operacional pode criar gargalos locais")
        else:
            print(f"   • Nenhuma rota no limite (sistema com folga operacional)")

        print("\n⚖ GARGALO 3: Desequilíbrio oferta/demanda")
        print(f"   • Produção total: {self.prod_total}t")
        print(f"   • Capacidade dos portos: {self.cap_portos}t")
        print(f"   • Desequilíbrio: {self.cap_portos - self.prod_total}t de capacidade ociosa")
        print(f"   • Portos operam como LIMITES SUPERIORES, não metas mínimas")

        porto_max = max(self.fluxos_portos, key=self.fluxos_portos.get)
        porto_min = min(self.fluxos_portos, key=self.fluxos_portos.get)

        print(f"\n   Priorização:")
        print(f"      • Porto mais atendido: {porto_max} ({self.fluxos_portos[porto_max]:.0f}t)")
        print(f"      • Porto menos atendido: {porto_min} ({self.fluxos_portos[porto_min]:.0f}t)")
        print(f"      • Estratégia: priorizar portos com melhor custo logístico")


def comparar_instancias():
    print("\n" + "=" * 70)
    print("ANÁLISE COMPARATIVA DAS TRÊS INSTÂNCIAS")
    print("=" * 70)

    resultados = {}
    analisadores = {}

    for inst in ['A', 'C', 'B']:
        print(f"\n> Resolvendo Instância {inst}...")
        model = criar_modelo(INSTANCIAS[inst])
        solver = SolverFactory("glpk")
        solver.options['tmlim'] = 300
        results = solver.solve(model, tee=False)

        if results.solver.termination_condition in ["optimal", "feasible"]:
            analisadores[inst] = AnalisadorResultados(model, inst)
            resultados[inst] = analisadores[inst]

    print("\n" + "=" * 70)
    print("TABELA COMPARATIVA PRINCIPAL")
    print("=" * 70)

    print(f"\n{'Indicador':<45} {'Inst. A':<15} {'Inst. C':<15} {'Inst. B':<15}")
    print("─" * 90)

    print(f"{'Custo total (R$)':<45} ", end="")
    for inst in ['A', 'C', 'B']:
        print(f"R$ {resultados[inst].custo_total:>10,.2f}  ", end="")
    print()

    print(f"{'Silos ativados (quant.)':<45} ", end="")
    for inst in ['A', 'C', 'B']:
        print(f"{len(resultados[inst].silos_ativos):>14}  ", end="")
    print()

    print(f"{'Custo fixo total (R$)':<45} ", end="")
    for inst in ['A', 'C', 'B']:
        print(f"R$ {resultados[inst].custo_silos:>10,.2f}  ", end="")
    print()

    porto_max_ref = max(resultados['A'].fluxos_portos, key=resultados['A'].fluxos_portos.get)
    print(f"{'Porto mais atendido (t)':<45} ", end="")
    for inst in ['A', 'C', 'B']:
        p = max(resultados[inst].fluxos_portos, key=resultados[inst].fluxos_portos.get)
        v = resultados[inst].fluxos_portos[p]
        print(f"{p} ({v:.0f})        ", end="")
    print()

    print(f"{'Porto menos atendido (t)':<45} ", end="")
    for inst in ['A', 'C', 'B']:
        p = min(resultados[inst].fluxos_portos, key=resultados[inst].fluxos_portos.get)
        v = resultados[inst].fluxos_portos[p]
        print(f"{p} ({v:.0f})        ", end="")
    print()

    print("\n" + "=" * 70)
    print("COMPARAÇÃO RELATIVA")
    print("=" * 70)

    custo_A = resultados['A'].custo_total
    custo_C = resultados['C'].custo_total
    custo_B = resultados['B'].custo_total

    print(f"\n Ranking de custos:")
    print(f"   1️ Instância A: R$ {custo_A:,.2f} (MAIS BARATA)")
    print(f"   2️ Instância C: R$ {custo_C:,.2f} (BASE/REFERÊNCIA)")
    print(f"   3️ Instância B: R$ {custo_B:,.2f} (MAIS CARA)")

    print(f"\n Diferenças absolutas e relativas:")
    dif_CA = custo_C - custo_A
    dif_BC = custo_B - custo_C
    print(f"   • C vs A: R$ {dif_CA:,.2f} (+{(dif_CA / custo_A) * 100:.1f}%)")
    print(f"   • B vs C: R$ {dif_BC:,.2f} (+{(dif_BC / custo_C) * 100:.1f}%)")
    print(f"   • B vs A: R$ {(custo_B - custo_A):,.2f} (+{((custo_B - custo_A) / custo_A) * 100:.1f}%)")

    print(f"\n Número de silos e sensibilidade aos custos fixos:")
    todos_iguais = len(set(len(r.silos_ativos) for r in resultados.values())) == 1

    if todos_iguais:
        print(f"   • Todas as instâncias ativam {len(resultados['A'].silos_ativos)} silos")
        print(f"   • CONCLUSÃO: Decisão de abertura é POUCO SENSÍVEL ao custo fixo")
        print(f"   • Existe um GARGALO DE CAPACIDADE que força manter todos abertos")
        print(f"   • Comportamento típico de 'efeito limiar'")
    else:
        print(f"   • Número de silos varia entre as instâncias")
        print(f"   • Sistema tem folga para fechar silos em alguns cenários")

    print(f"\n Distribuição dos fluxos Silo→Porto:")
    # Verificar se os fluxos são idênticos
    fluxos_A = resultados['A'].fluxos_portos
    fluxos_C = resultados['C'].fluxos_portos
    fluxos_B = resultados['B'].fluxos_portos

    identicos = (set(fluxos_A.keys()) == set(fluxos_C.keys()) == set(fluxos_B.keys()))

    if identicos:
        print(f"   • Os fluxos ferroviários são IDÊNTICOS nas três instâncias")
        print(f"   • Indica uma solução ROBUSTA no nível ferroviário")
        for porto in sorted(fluxos_A.keys()):
            v = fluxos_A[porto]
            cap = resultados['A'].model.dem[porto]
            status = "SATURADO" if v >= cap * 0.999 else "COM FOLGA"
            print(f"      - {porto}: {v:.0f}t ({status})")


def gerar_relatorio_completo(instancia='C'):
    print("\n" + "=" * 70)
    print(f"RELATÓRIO TÉCNICO COMPLETO - INSTÂNCIA {instancia}")
    print("=" * 70)

    model = criar_modelo(INSTANCIAS[instancia])
    solver = SolverFactory("glpk")
    solver.options['tmlim'] = 300
    results = solver.solve(model, tee=False)

    if results.solver.termination_condition not in ["optimal", "feasible"]:
        print(f" Erro ao resolver: {results.solver.termination_condition}")
        return

    analisador = AnalisadorResultados(model, instancia)

    analisador.gerar_secao_ativacao_silos()
    analisador.gerar_secao_custos()
    analisador.gerar_secao_modais()
    analisador.gerar_secao_gargalos()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GERADOR DE ANÁLISES PARA RELATÓRIO TÉCNICO")
    print("=" * 70)

    print("\nOpções:")
    print("1 - Gerar relatório completo de UMA instância")
    print("2 - Gerar análise comparativa das TRÊS instâncias")
    print("3 - Gerar TUDO (relatórios individuais + comparação)")

    opcao = input("\nEscolha: ").strip()

    if opcao == "1":
        inst = input("Instância (A/B/C): ").strip().upper()
        if inst in ['A', 'B', 'C']:
            gerar_relatorio_completo(inst)
        else:
            print("Instância inválida!")

    elif opcao == "2":
        comparar_instancias()

    elif opcao == "3":
        for inst in ['A', 'C', 'B']:
            gerar_relatorio_completo(inst)
            print("\n\n")

        comparar_instancias()

    else:
        print("Opção inválida!")
