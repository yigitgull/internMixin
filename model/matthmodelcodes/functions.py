import numpy as np
import pyomo.environ as pyo
import pandas as pd
import os
import datetime
from gurobipy import GRB
from numpy import dot, array

eps = 1e-4


def get_j_set(m):
    return pyo.Set(initialize=[(j, f) for f in m.f for j in m.j[f]])


def get_t_set(m):
    return pyo.Set(initialize=[(t, mc) for mc in m.mc for t in m.t[mc]])


def get_job_data(m, data, col, three_d=False):
    if not three_d:
        return {(j, f): data[data["Type"] == f].loc[j, col] for f in m.f for j in m.j[f]}
    else:
        return {(j, f, mc): data[data["Type"] == f].loc[j, col] * m.RC[mc]
                for f in m.f for j in m.j[f] for mc in m.mc}


def objective_function_weights(objective_funcs_list, weights_list):
    for i, weight in enumerate(weights_list):
        if weight == 0:
            weights_list[i] = 1e-2

    obj = 0
    for i, (weight, objective) in enumerate(zip(weights_list, objective_funcs_list)):
        obj += weight * objective

    return obj


def print_vars(writer, one_dims, two_dims, three_dims):
    for pyo_var in one_dims:
        df = pd.DataFrame.from_dict(pyo_var.extract_values(), orient="index")
        df.to_excel(writer, sheet_name=pyo_var.name)
    for pyo_var in two_dims:
        dct = pyo_var.extract_values()
        row_num = 0
        col_num = 0
        for index_tuple in dct.keys():
            if row_num < index_tuple[0]:
                row_num = index_tuple[0]
            if col_num < index_tuple[1]:
                col_num = index_tuple[1]
        df = pd.DataFrame([[None] * col_num] * row_num, columns=[k for k in range(1, col_num + 1)],
                          index=[j for j in range(1, row_num + 1)])
        for ((j, k), value) in dct.items():
            df.iloc[j - 1, k - 1] = value
        df.to_excel(writer, sheet_name=pyo_var.name)
    for pyo_var in three_dims:
        dct = pyo_var.extract_values()
        col_num = 0
        df_index = []
        for index_tuple in dct.keys():
            df_index.append((index_tuple[1], index_tuple[2]))
            if col_num < index_tuple[0]:
                col_num = index_tuple[0]
        df = pd.DataFrame([[None] * col_num] * len(df_index), index=pd.MultiIndex.from_tuples(df_index))
        for ((j, k, l), value) in dct.items():
            df.iloc[[k - 1, l - 1], j - 1] = value
        df.sort_index(inplace=True)
        df.to_excel(writer, sheet_name=pyo_var.name)


def print_objectives(m, writer, callback_object):
    other_objs_dct = {"Tardiness": [m._tardiness],
                      "W-Tardiness": [m._weighted_tardiness],
                      "Num of Tardies": [m._num_of_tardies],
                      "W-Num of Tardies": [m._weighted_num_of_tardies],
                      "Max_Tardiness": [m._max_tardiness],
                      "W-Max_Tardiness": [m._weighted_max_tardiness],
                      "Time Taken": callback_object.time_taken if callback_object.time_taken > 1 else (datetime.datetime.now() - callback_object.time).seconds,
                      "Node Count": callback_object.node_count}
    df = pd.DataFrame(other_objs_dct)
    df.to_excel(writer, sheet_name="All Objectives", index=True)


def print_results(file_path, file_name, m, one_dims=(), two_dims=(), three_dims=(), callback_object=None):
    path = os.path.join(file_path, file_name)

    writer = pd.ExcelWriter(path)
    print_vars(writer, one_dims=one_dims, two_dims=two_dims, three_dims=three_dims)
    print_objectives(m, writer, callback_object=callback_object)
    writer.close()


def summarize_results(main_directory, summary_file):
    # Define the sheet name you want to extract data from
    sheet_name = 'All Objectives'

    # Initialize an empty DataFrame to store the data
    combined_data = pd.DataFrame()

    # Walk through the main directory and its subfolders
    for root, dirs, files in os.walk(main_directory):
        # print(root, dirs, files)
        for file in files:
            if file.endswith('.xlsx'):
                file_path = os.path.join(root, file)
                try:
                    # Read the Excel file into a DataFrame
                    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
                    df.index = [root.replace("Outputs\\", "") + " - " + file.replace(".xlsx", "")]

                    # Append the data to the combined DataFrame
                    combined_data = pd.concat([df, combined_data])
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    combined_data.sort_index(inplace=True)
    combined_data.to_excel(summary_file, sheet_name="Summary")


def solve_and_print(m, obj, opt, file_path, file_name, cb, one_dims=(), two_dims=(), three_dims=(), schedule=lambda x: x):
    obj.activate()

    opt.set_instance(m)
    opt.solve(tee=True)

    time = 0
    C = {}

    schedule = schedule(m)

    for j in schedule:
        time += m.p[j]
        C[j] = time

    m._C = pyo.Param(m.J, initialize=lambda m, j: C[j])

    m._tardiness = sum([max(0, pyo.value(m._C[j]) - m.d[j]) for j in m.J])
    m._weighted_tardiness = sum([m.w[j] * max(0, pyo.value(m._C[j]) - m.d[j]) for j in m.J])

    m._num_of_tardies = sum([1 if pyo.value(m._C[j]) - m.d[j] > eps else 0 for j in m.J])
    m._weighted_num_of_tardies = sum([m.w[j] if pyo.value(m._C[j]) - m.d[j] > eps else 0 for j in m.J])

    m._max_tardiness = max([max(0, pyo.value(m._C[j]) - m.d[j]) for j in m.J])
    m._weighted_max_tardiness = max([m.w[j] * max(0, pyo.value(m._C[j]) - m.d[j]) for j in m.J])

    print_results(file_path, file_name, m,
                  one_dims=one_dims,
                  two_dims=two_dims,
                  three_dims=three_dims,
                  callback_object=cb)
    cb.exit(opt)
    cb.reset()
    obj.deactivate()


class CallbackFunction:
    best_obj = float("inf")
    time_since_best = 0
    time_limit = 0
    internal_counter = 0
    counter_limit = 0
    best_bound = float("inf")
    tolerance = 0
    time_taken = 0
    node_count = 0
    time = datetime.datetime.now()
    terminated = False

    def __init__(self, time_limit, counter_limit=5, tolerance=1):
        self.time_limit = time_limit
        self.counter_limit = counter_limit
        self.tolerance = tolerance

    def callback(self, cb_m, cb_opt, cb_where):
        if cb_where == 5:
            # Get model objective
            obj = cb_opt.cbGet(GRB.Callback.MIPNODE_OBJBST)
            best_bd = cb_opt.cbGet(GRB.Callback.MIPNODE_OBJBND)

            # Has objective changed?
            if abs(obj - self.best_obj) > self.tolerance or abs(best_bd - self.best_bound) > self.tolerance:
                # If so, update incumbent and time
                self.best_obj = obj
                self.best_bound = best_bd
                self.time_since_best = datetime.datetime.now()
                self.internal_counter = 0
            else:
                self.internal_counter += 1

            self.node_count = cb_opt.cbGet(GRB.Callback.MIPNODE_NODCNT)

            # Terminate if objective has not improved in 20s
            if ((datetime.datetime.now() - self.time_since_best).seconds > self.time_limit and
                    self.internal_counter > self.counter_limit):
                self.terminated = True
                cb_opt._solver_model.terminate()

    def exit(self, cb_opt):
        if self.terminated:
            self.time_taken = (datetime.datetime.now() - self.time).seconds - self.time_limit
        else:
            self.time_taken = (datetime.datetime.now() - self.time).seconds

    def reset(self):
        self.time_since_best = 0
        self.best_obj = float("inf")
        self.best_bound = float("inf")
        self.time_taken = 0
        self.node_count = 0
        self.internal_counter = 0
        self.time = datetime.datetime.now()
        self.terminated = False
