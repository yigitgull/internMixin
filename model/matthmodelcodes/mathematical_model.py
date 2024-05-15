import pyomo.opt as pyopt
import pyomo.environ as pyo
import pandas as pd
import functions as funcs


def main(time_limit=90, counter_limit=30, tolerance=1, focus=0, termination_gap=0):
    bigM = 1e9
    deadline = 24 * 2

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 8000)

    major_set_up = pd.read_csv("major_setup_times.csv", header=0, index_col=0)
    relevant_data = pd.read_csv("relevant_data.csv", header=0, index_col=0)

    m = pyo.ConcreteModel(name="Mathematical Model")
    m.f = pyo.Set(initialize=major_set_up.columns.astype(int).to_list())
    m.j = pyo.Set(m.f, initialize={f: relevant_data[relevant_data["Type"] == f].index.to_list() for f in m.f})
    m.jf = funcs.get_j_set(m)

    m.mc = pyo.Set(initialize=range(1, 3 + 1))
    m.t = pyo.Set(m.mc, initialize={mc: list(range(1, 8 + 1)) for mc in m.mc})
    m.tmc = funcs.get_t_set(m)

    m.w = pyo.Param(m.jf,
                    initialize=funcs.get_job_data(m, relevant_data, "Weight"))
    m.P_minor = pyo.Param(m.jf,
                          initialize=funcs.get_job_data(m, relevant_data, "Minor Setup Time"))
    m.P_major = pyo.Param(m.f, m.f, initialize={(f, g): major_set_up.loc[f, str(g)] for f in m.f for g in m.f})
    m.RC = pyo.Param(m.mc, initialize={mc: v for mc, v in zip(m.mc, [1, 0.95, 0.9])})
    m.Capacity = pyo.Param(m.jf, m.mc,
                           initialize=funcs.get_job_data(m, relevant_data, "Capacity of Line 1", True))
    m.Demand = pyo.Param(m.jf, initialize=funcs.get_job_data(m, relevant_data, "Demand"))
    m.Labor = pyo.Param(m.jf,
                        initialize=funcs.get_job_data(m, relevant_data, "Number of Labor Required"))

    m.x = pyo.Var(m.jf, m.tmc, initialize=0, domain=pyo.Binary)
    m.y = pyo.Var(m.f, m.f, m.tmc, initialize=0, domain=pyo.Binary)
    m.c_jt = pyo.Var(m.jf, m.tmc, initialize=0, domain=pyo.NonNegativeReals)
    m.c_j = pyo.Var(m.jf, initialize=0, domain=pyo.NonNegativeReals)

    m.c_max = pyo.Var(domain=pyo.NonNegativeReals)
    m.flow_time = pyo.Var(domain=pyo.NonNegativeReals)
    m.w_flow_time = pyo.Var(domain=pyo.NonNegativeReals)
    m.tardiness_quantity = pyo.Var(m.jf, domain=pyo.NonNegativeReals)
    m.tardiness = pyo.Var(domain=pyo.NonNegativeReals)
    m.w_tardiness = pyo.Var(domain=pyo.NonNegativeReals)
    m.tardy_binary = pyo.Var(m.jf, domain=pyo.Binary, initialize=0)
    m.tardy_jobs = pyo.Var(domain=pyo.NonNegativeReals)
    m.w_tardy_jobs = pyo.Var(domain=pyo.NonNegativeReals)

    def job_position_constraint(mo, j, f):
        return sum([m.x[(j, f, t, mc)] for (t, mc) in m.tmc]) == 1
    m.job_position_constraint = pyo.Constraint(m.jf, rule=job_position_constraint)

    def position_job_constraint(mo, t, mc):
        return sum([m.x[(j, f, t, mc)] for (j, f) in m.jf]) <= 1
    m.position_job_constraint = pyo.Constraint(m.tmc, rule=position_job_constraint)

    def completion_time_constraint(mo, j, f):
        return m.c_j[(j, f)] == sum([m.c_jt[(j, f, t, mc)] for (t, mc) in m.tmc])
    m.completion_time_constraint = pyo.Constraint(m.jf, rule=completion_time_constraint)

    def completion_time_jt_constraint(mo, j, f, t, mc):
        if t > 1:
            return m.c_jt[(j, f, t, mc)] >= (sum([m.c_jt[(i, g, t - 1, mc)] for (i, g) in m.jf]) +
                                             m.P_minor[(j, f)] +
                                             sum([m.P_major[f, g] * m.y[(f, g, t, mc)] for g in m.f]) +
                                             m.Demand[(j, f)] / m.Capacity[(j, f, mc)] -
                                             bigM * (1 - m.x[j, f, t, mc]))
        else:
            return m.c_jt[(j, f, t, mc)] >= (m.P_minor[(j, f)] +
                                             sum([m.P_major[f, g] * m.y[(f, g, t, mc)] for g in m.f]) +
                                             m.Demand[(j, f)] / m.Capacity[(j, f, mc)] -
                                             bigM * (1 - m.x[j, f, t, mc]))
    m.completion_time_jt_constraint = pyo.Constraint(m.jf, m.tmc,
                                                     rule=completion_time_jt_constraint)

    def major_setup_constraint(mo, f, g, t, mc):
        if t > 1 and f != g:
            return m.y[f, g, t, mc] >= sum([m.x[i, g, t - 1, mc] for i in m.j[g]])
        else:
            return pyo.Constraint.Feasible
    m.major_setup_constraint = pyo.Constraint(m.f, m.f, m.tmc, rule=major_setup_constraint)

    def c_max_constraint(mo, j, f):
        return m.c_max >= m.c_j[(j, f)]
    m.c_max_constraint = pyo.Constraint(m.jf, rule=c_max_constraint)

    def flow_time_constraint(mo):
        return m.flow_time >= sum([m.c_j[(j, f)] for j, f in m.jf])
    m.flow_time_constraint = pyo.Constraint(rule=flow_time_constraint)

    def w_flow_time_constraint(mo):
        return m.w_flow_time >= sum([m.w[(j, f)] * m.c_j[(j, f)] for j, f in m.jf])
    m.w_flow_time_constraint = pyo.Constraint(rule=w_flow_time_constraint)

    def tardiness_quantity_constraint(mo, j, f):
        return m.tardiness_quantity[(j, f)] >= m.c_j[(j, f)] - deadline
    m.tardiness_quantity_constraint = pyo.Constraint(m.jf, rule=tardiness_quantity_constraint)

    def tardiness_constraint(mo):
        return m.tardiness >= sum([m.tardiness_quantity[(j, f)] for j, f in m.jf])
    m.tardiness_constraint = pyo.Constraint(rule=tardiness_constraint)

    def w_tardiness_constraint(mo):
        return m.w_tardiness >= sum([m.w[(j, f)] * m.tardiness_quantity[(j, f)] for j, f in m.jf])
    m.w_tardiness_constraint = pyo.Constraint(rule=w_tardiness_constraint)

    def tardy_constraint(mo, j, f):
        return bigM * m.tardy_binary[(j, f)] >= m.tardiness_quantity[(j, f)]
    m.tardy_constraint = pyo.Constraint(m.jf, rule=tardy_constraint)

    def tardy_jobs_constraint(mo):
        return m.tardy_jobs >= sum([m.tardy_binary[(j, f)] for (j, f) in m.jf])
    m.tardy_jobs_constraint = pyo.Constraint(rule=tardy_jobs_constraint)

    def w_tardy_jobs_constraint(mo):
        return m.w_tardy_jobs >= sum([m.w[(j, f)] * m.tardy_binary[(j, f)] for (j, f) in m.jf])
    m.w_tardy_jobs_constraint = pyo.Constraint(rule=w_tardy_jobs_constraint)

    callback = funcs.CallbackFunction(time_limit=time_limit, counter_limit=counter_limit, tolerance=tolerance)
    opt = pyopt.SolverFactory('gurobi_persistent', solver_io="python")
    opt.options["MIPFocus"] = focus
    opt.options["MIPGap"] = termination_gap
    opt.set_callback(callback.callback)

    def solve_model(_model, weights, name):
        model = _model.clone()

        objectives_list = [model.c_max, model.flow_time, model.w_flow_time,
                           model.tardiness, model.w_tardiness,
                           model.tardy_jobs, model.w_tardy_jobs]

        model.objective = pyo.Objective(rule=funcs.objective_function_weights(objectives_list, weights),
                                        sense=pyo.minimize, name=name)

        opt.set_instance(model)
        print(opt._solver_model)
        opt.solve(tee=True)
        print([pyo.value(x) for x in objectives_list])
        del model

    solve_model(m, [1, 1, 0, 0, 0, 0, 0], "Test")
