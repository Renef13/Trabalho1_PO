from msilib import Binary

from pyomo.environ import *


def criar_modelo(dados):
    model = ConcreteModel(name="Transbordo_Bacuri")

    model.F = Set(initialize=dados["prod"].keys(), doc="Fazendas")
    model.S = Set(initialize=dados["cap_silo"].keys(), doc="Silos")
    model.P = Set(initialize=dados["dem"].keys(), doc="Portos")

    model.Arcos_FS = Set(initialize=model.F * model.S, doc="Arcos Fazenda-Silo")
    model.Arcos_SP = Set(initialize=model.S * model.P, doc="Arcos Silo-Porto")

    model.prod = Param(model.F, initialize=dados["prod"], doc="Produção das fazendas (t)")

    model.dem = Param(model.P, initialize=dados["dem"], doc="Demanda dos portos (t)")

    model.cap_silo = Param(model.S, initialize=dados["cap_silo"], doc="Capacidade dos silos (t)")
    model.cap_carr = Param(initialize=dados["cap_carr"], doc="Capacidade por carroceria (t)")

    model.cf_silo = Param(model.S, initialize=dados["cf_silo"], doc="Custo fixo de abertura do silo ($)")
    model.cf_carr = Param(initialize=dados["cf_carr"], doc="Custo fixo por carroceria ($)")

    model.custo_ferro = Param(model.Arcos_SP, initialize=dados["custo_ferro"],
                              doc="Custo de transporte ferroviário ($/t)")

    M = sum(dados["prod"].values())
    model.M = Param(initialize=M, doc="Big-M para restrições condicionais")

    model.x = Var(model.Arcos_FS, within=NonNegativeReals, doc="Fluxo Fazenda-Silo (t)")
    model.y = Var(model.Arcos_SP, within=NonNegativeReals, doc="Fluxo Silo-Porto (t)")

    model.z = Var(model.S, within=Binary, doc="Ativação do silo (0/1)")
    model.w = Var(model.P, within=Binary, doc="Ativação do porto (0/1)")

    model.t = Var(model.Arcos_FS, within=NonNegativeIntegers, bounds=(0, 3), doc="Número de carrocerias (0-3)")

    def objetivo_rule(m):
        custo_silos = sum(m.cf_silo[j] * m.z[j] for j in m.S)
        custo_carrocerias = sum(m.cf_carr * m.t[i, j] for (i, j) in m.Arcos_FS)
        custo_transporte = sum(m.custo_ferro[j, k] * m.y[j, k] for (j, k) in m.Arcos_SP)

        return custo_silos + custo_carrocerias + custo_transporte

    model.objetivo = Objective(rule=objetivo_rule, sense=minimize, doc="Minimizar custo total")

    def conservacao_fazenda_rule(m, i):
        return sum(m.x[i, j] for j in m.S) == m.prod[i]

    model.conservacao_fazenda = Constraint(model.F, rule=conservacao_fazenda_rule,
                                           doc="R1: Conservação de fluxo nas fazendas")

    def conservacao_silo_rule(m, j):
        entrada = sum(m.x[i, j] for i in m.F)
        saida = sum(m.y[j, k] for k in m.P)
        return entrada == saida

    model.conservacao_silo = Constraint(model.S, rule=conservacao_silo_rule,
                                        doc="R2: Conservação de fluxo nos silos")

    def demanda_minima_rule(m, k):
        return m.dem[k] * m.w[k] <= sum(m.y[j, k] for j in m.S)

    model.demanda_minima = Constraint(model.P, rule=demanda_minima_rule,
                                      doc="R3a: Demanda mínima dos portos ativos")

    def demanda_maxima_rule(m, k):
        return sum(m.y[j, k] for j in m.S) <= m.M * m.w[k]

    model.demanda_maxima = Constraint(model.P, rule=demanda_maxima_rule,
                                      doc="R3b: Porto inativo não recebe carga")

    def capacidade_silo_rule(m, j):
        return sum(m.x[i, j] for i in m.F) <= m.cap_silo[j] * m.z[j]

    model.capacidade_silo = Constraint(model.S, rule=capacidade_silo_rule,
                                       doc="R4: Capacidade dos silos")

    def capacidade_carroceria_rule(m, i, j):
        return m.x[i, j] <= m.cap_carr * m.t[i, j]

    model.capacidade_carroceria = Constraint(model.Arcos_FS, rule=capacidade_carroceria_rule,
                                             doc="R5: Capacidade das carrocerias")

    return model
