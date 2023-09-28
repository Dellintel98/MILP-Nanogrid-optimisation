import pandas as pd
import os
import pyomo
import pyomo.opt
import pyomo.environ as pe
from pyomo.core import Var
import logging
import numpy as np
import datetime


class Nanogrid:
    def __init__(self, components_data, parking_lot_data, market_data, pv_system_data, building_data,
                 model_variant, end_soe_equal_to_requested_soe):
        self.components_data = components_data
        self.parking_lot_data = parking_lot_data
        self.market_data = market_data
        self.pv_system_data = pv_system_data
        self.building_data = building_data

        self.dt = 0.25

        self.time_set = np.arange(1, parking_lot_data.shape[0] + 1, 1)
        self.month_set = np.arange(1, 12 + 1, 1)
        self.quarter_set = np.arange(1, 4 + 1, 1)
        self.year_set = np.arange(1, int(self.components_data.loc['FP_lifetime', 'VALUE']) + 1, 1)
        self.loan_set = np.arange(1, int(self.components_data.loc['FP_payback_time', 'VALUE']) + 1, 1)
        self.parking_lot_set = parking_lot_data.columns.levels[0]

        self.end_soe_equal_to_requested_soe = end_soe_equal_to_requested_soe

        self.model = None
        self.model_variant = model_variant
        self.create_model()

    def create_model(self):
        print('Building model')
        self.model = pe.ConcreteModel()

        self.create_sets()
        self.create_parameters()
        self.create_variables()

        self.create_objective()
        self.create_constraints()

        print('Checkpoint 15: Model successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_sets(self):
        self.model.time_set = pe.Set(initialize=self.time_set)
        self.model.month_set = pe.Set(initialize=self.month_set)
        self.model.quarter_set = pe.Set(initialize=self.quarter_set)
        self.model.year_set = pe.Set(initialize=self.year_set)
        self.model.loan_set = pe.Set(initialize=self.loan_set)
        self.model.parking_lot_set = pe.Set(initialize=self.parking_lot_set)
        self.model.components_set = pe.Set(initialize=self.components_data.index.to_list())
        self.model.market_set = pe.Set(initialize=self.market_data.columns)

        print('Checkpoint 01: Sets successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_parameters(self):
        self.model.components_data = pe.Param(self.model.components_set, mutable=True,
                                              initialize=self.components_data['VALUE'].to_dict())
        self.model.pv_system_data = pe.Param(self.model.time_set, initialize=self.pv_system_data['PV'].to_dict())

        # Building data - Object Demand (OD) data
        self.model.P_OD_data = self.load_building_data('2016')

        parking_lot_params = self.parking_lot_data.columns.levels[1].to_list()
        idx = pd.IndexSlice
        parking_lots = {}
        for p in parking_lot_params:
            parking_lots[p] = self.parking_lot_data.loc[idx[:], idx[:, p]].droplevel(level=1, axis=1).stack().to_dict()

        self.model.PL_ev_arrival = pe.Param(self.model.time_set, self.model.parking_lot_set,
                                            initialize=parking_lots['arrival'], default=0)
        self.model.PL_ev_available = pe.Param(self.model.time_set, self.model.parking_lot_set,
                                              initialize=parking_lots['available'], default=0)
        self.model.PL_ev_departure = pe.Param(self.model.time_set, self.model.parking_lot_set,
                                              initialize=parking_lots['departure'], default=0)
        self.model.EV_required_end_state = pe.Param(self.model.time_set, self.model.parking_lot_set,
                                                    initialize=parking_lots['end'], default=0)
        self.model.EV_state_on_arrival = pe.Param(self.model.time_set, self.model.parking_lot_set,
                                                  initialize=parking_lots['initial'], default=0)
        self.model.E_ev_capacity = pe.Param(self.model.time_set, self.model.parking_lot_set,
                                            initialize=parking_lots['capacity'], default=0)

        self.model.market_data = pe.Param(self.model.time_set, self.model.market_set,
                                          initialize=self.market_data.stack().to_dict(), default=0)
        # self.model.pen = 0.2 / 40
        if self.model_variant == 1 or self.model_variant == 2:
            self.model.pen = 0.2 / 280
        else:
            self.model.pen = 0.2 / 80
        # self.model.pen = 0.0
        print('Checkpoint 02: Parameters successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def load_building_data(self, data_year):
        return pe.Param(self.model.time_set, initialize=self.building_data[data_year].to_dict())

    def create_variables(self):
        self.set_pv_variables()
        self.set_battery_variables()
        self.set_grid_variables()
        self.set_parking_lot_variables()
        self.set_costs_variables()

        print('Checkpoint 03: Variables successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # noinspection PyUnresolvedReferences
    def set_pv_variables(self):
        self.model.P_pv_install = pe.Var(domain=pe.NonNegativeReals)
        self.model.P_pv = pe.Var(self.model.time_set, domain=pe.NonNegativeReals)
        self.model.binary_pv = pe.Var(domain=pe.Binary)

    # noinspection PyUnresolvedReferences
    def set_battery_variables(self):
        self.model.E_battery_capacity = pe.Var(domain=pe.NonNegativeReals)
        self.model.binary_battery = pe.Var(domain=pe.Binary)
        self.model.E_battery = pe.Var(self.model.time_set, domain=pe.NonNegativeReals)
        self.model.P_battery_ch = pe.Var(self.model.time_set, domain=pe.NonNegativeReals)
        self.model.P_battery_ds = pe.Var(self.model.time_set, domain=pe.NonNegativeReals)
        self.model.P_battery_MAX = pe.Var(domain=pe.NonNegativeReals)
        self.model.P_battery_max = pe.Var(self.model.time_set, domain=pe.NonNegativeReals)
        self.model.P_battery_max_ch = pe.Var(self.model.time_set, domain=pe.NonNegativeReals)
        self.model.Bat_ch_ramp = pe.Var(self.model.time_set, domain=pe.NonNegativeReals)

    # noinspection PyUnresolvedReferences
    def set_grid_variables(self):
        self.model.P_grid = pe.Var(self.model.time_set, domain=pe.Reals)
        self.model.P_grid_positive = pe.Var(self.model.time_set, domain=pe.NonNegativeReals)
        self.model.P_grid_negative = pe.Var(self.model.time_set, domain=pe.NonNegativeReals)
        self.model.P_grid_max = pe.Var(self.model.month_set, domain=pe.Reals)
        self.model.P_contracted = pe.Var(domain=pe.NonNegativeReals)
        self.model.P_cs_contracted = pe.Var(domain=pe.NonNegativeReals)

    # noinspection PyUnresolvedReferences
    def set_parking_lot_variables(self):
        self.model.P_ev_ch = pe.Var(self.model.parking_lot_set, self.model.time_set,
                                    domain=pe.NonNegativeReals)
        self.model.P_ev_ds = pe.Var(self.model.parking_lot_set, self.model.time_set,
                                    domain=pe.NonNegativeReals)
        self.model.E_ev = pe.Var(self.model.parking_lot_set, self.model.time_set,
                                 domain=pe.NonNegativeReals)
        self.model.SOE_ev_relative = pe.Var(self.model.parking_lot_set, self.model.time_set,
                                            domain=pe.NonNegativeReals, bounds=(0, 1))
        self.model.EV_ch_ramp = pe.Var(self.model.parking_lot_set, self.model.time_set, domain=pe.NonNegativeReals)

    # noinspection PyUnresolvedReferences
    def set_costs_variables(self):
        self.model.C_total = pe.Var(domain=pe.Reals)
        self.model.C_invest = pe.Var(domain=pe.NonNegativeReals)
        self.model.C_ee_operational = pe.Var(domain=pe.NonNegativeReals)
        self.model.C_ee_annual = pe.Var(domain=pe.NonNegativeReals)
        self.model.C_ee_hourly = pe.Var(self.model.time_set, domain=pe.NonNegativeReals)
        self.model.C_profit_annual = pe.Var(domain=pe.NonNegativeReals)
        self.model.C_profit_hourly = pe.Var(self.model.time_set, domain=pe.NonNegativeReals)
        self.model.C_profit = pe.Var(domain=pe.NonNegativeReals)
        self.model.C_maintenance = pe.Var(domain=pe.NonNegativeReals)
        self.model.C_pv_maintenance = pe.Var(domain=pe.NonNegativeReals)
        self.model.C_battery_maintenance = pe.Var(domain=pe.NonNegativeReals)
        self.model.C_pl_maintenance = pe.Var(domain=pe.NonNegativeReals)

        self.model.C_variations_annual = pe.Var(domain=pe.NonNegativeReals)
        self.model.C_variations = pe.Var(domain=pe.NonNegativeReals)

        self.model.C_loan = pe.Var(domain=pe.NonNegativeReals)
        self.model.C_annuity = pe.Var(domain=pe.NonNegativeReals)
        self.model.C_battery_replacement = pe.Var(domain=pe.NonNegativeReals)

    def create_objective(self):
        def objective(model):
            return model.C_total + model.C_variations

        self.model.objective = pe.Objective(rule=objective, sense=pe.minimize)

        def variations_annual(model):
            return model.C_variations_annual == (
                        sum(model.EV_ch_ramp[lot, t] for t in model.time_set for lot in model.parking_lot_set) +
                        sum(model.Bat_ch_ramp[t] for t in model.time_set))

        self.model.variations_annual = pe.Constraint(rule=variations_annual)

        def variations_penalization(model):
            return model.C_variations == sum(
                model.pen * model.C_variations_annual / (1 + model.components_data["FP_discount_rate"]) ** year for year
                in model.year_set)

        self.model.variations_penalization = pe.Constraint(rule=variations_penalization)

        def total_costs(model):
            return model.C_total == model.C_invest + model.C_loan + model.C_maintenance + \
                   model.C_ee_operational + model.C_battery_replacement - model.C_profit

        self.model.total_costs = pe.Constraint(rule=total_costs)

        print('Checkpoint 04: Objective function '
              'successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_constraints(self):
        self.create_investment_and_loan_costs_constraints()
        self.create_equipment_maintenance_and_replacement_costs_constraints()
        self.create_operational_costs_constraints()
        self.create_profit_constraints()

        self.create_power_balance_equation()

        self.create_grid_constraints()
        self.create_pv_system_constraints()
        self.create_battery_system_constraints()
        self.create_electric_vehicle_constraints()

        print('Checkpoint 14: Constraints successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_investment_and_loan_costs_constraints(self):
        def total_investment_costs(model):
            return model.C_invest == (model.components_data["PL_var_cost"] * model.components_data["PL_Nb_lots"] +
                                      model.components_data["SP_var_cost"] * model.P_pv_install +
                                      model.components_data["BS_var_cost_W"] * model.E_battery_capacity +
                                      model.components_data["GC_var_cost"] * model.P_cs_contracted) \
                   * (1 - model.components_data["FP_loan_ratio"])

        self.model.total_investment_costs = pe.Constraint(rule=total_investment_costs)

        def annuity_costs(model):
            return model.C_annuity == (model.components_data["PL_var_cost"] * model.components_data["PL_Nb_lots"] +
                                       model.components_data["SP_var_cost"] * model.P_pv_install +
                                       model.components_data["BS_var_cost_W"] * model.E_battery_capacity +
                                       model.components_data["GC_var_cost"] * model.P_cs_contracted) \
                   * model.components_data["FP_loan_ratio"] \
                   * model.components_data["FP_interest_rate"] / \
                   (1 - (1 +
                         model.components_data["FP_interest_rate"]) ** (-1 * model.components_data["FP_payback_time"]))

        self.model.annuity_costs = pe.Constraint(rule=annuity_costs)

        def loan(model):
            return model.C_loan == sum(model.C_annuity / (1 + model.components_data["FP_discount_rate"]) ** year
                                       for year in model.loan_set)

        self.model.loan = pe.Constraint(rule=loan)

        print('Checkpoint 05: Investment and loan costs'
              ' constraints successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_equipment_maintenance_and_replacement_costs_constraints(self):
        def pv_system_maintenance_costs(model):
            return model.C_pv_maintenance == model.components_data["SP_var_cost"] * model.P_pv_install \
                   * model.components_data["SP_oper_cost"]

        self.model.pv_system_maintenance_costs = pe.Constraint(rule=pv_system_maintenance_costs)

        def battery_system_maintenance_costs(model):
            return model.C_battery_maintenance == model.components_data["BS_var_cost_W"] * model.E_battery_capacity \
                   * model.components_data["BS_oper_cost"]

        self.model.battery_system_maintenance_costs = pe.Constraint(rule=battery_system_maintenance_costs)

        def parking_lot_maintenance_costs(model):
            return model.C_pl_maintenance == model.components_data["PL_var_cost"] * model.components_data["PL_Nb_lots"] \
                   * model.components_data["PL_oper_cost"]

        self.model.parking_lot_maintenance_costs = pe.Constraint(rule=parking_lot_maintenance_costs)

        def maintenance_costs(model):
            return model.C_maintenance == sum(
                ((model.C_pv_maintenance + model.C_battery_maintenance + model.C_pl_maintenance) /
                 (1 + model.components_data["FP_discount_rate"]) ** year) for year in model.year_set)

        self.model.maintenance_costs = pe.Constraint(rule=maintenance_costs)

        def battery_system_replacement_costs(model):
            return model.C_battery_replacement == model.components_data["BS_var_cost_W"] * model.components_data[
                "BS_replacement_perc"] * model.E_battery_capacity / \
                   (1 + model.components_data["FP_discount_rate"]) ** model.components_data["BS_replacement_year"]

        self.model.battery_system_replacement_costs = pe.Constraint(rule=battery_system_replacement_costs)

        print('Checkpoint 06: Equipment maintenance and replacement costs'
              ' constraints successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_operational_costs_constraints(self):
        def total_operational_electricity_costs(model):
            return model.C_ee_operational == sum((model.C_ee_annual * (1 + model.components_data["EP_annual_growth"])
                                                  ** year / (1 + model.components_data["FP_discount_rate"]) ** year)
                                                 for year in model.year_set)

        self.model.total_operational_electricity_costs = pe.Constraint(rule=total_operational_electricity_costs)

        def annual_electricity_costs(model):
            return model.C_ee_annual == sum(model.C_ee_hourly[t] for t in model.time_set) + \
                   model.components_data["GT_Peak_cost"] * sum(model.P_grid_max[month] for month in model.month_set)

        self.model.annual_electricity_costs = pe.Constraint(rule=annual_electricity_costs)

        def hourly_electricity_costs(model, t):
            if model.market_data[t, 'tariff'] == 1:
                c_ee_tariff = (model.components_data["EP_HT_cost"] +
                               model.components_data["GT_HT_cost"] +
                               model.components_data["GT_RES_incentive"])
            else:
                c_ee_tariff = (model.components_data["EP_LT_cost"] +
                               model.components_data["GT_LT_cost"] +
                               model.components_data["GT_RES_incentive"])
            return model.C_ee_hourly[t] == c_ee_tariff * model.P_grid_positive[t] * self.dt

        self.model.hourly_electricity_costs = pe.Constraint(self.model.time_set, rule=hourly_electricity_costs)

        print('Checkpoint 07: Operational costs'
              ' constraints successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_profit_constraints(self):
        def total_profit(model):
            return model.C_profit == sum((model.C_profit_annual *
                                          (1 + model.components_data["EP_annual_growth"]) ** year /
                                          (1 + model.components_data["FP_discount_rate"]) ** year)
                                         for year in model.year_set)

        self.model.total_profit = pe.Constraint(rule=total_profit)

        def annual_profit(model):
            return model.C_profit_annual == sum(model.C_profit_hourly[t] for t in model.time_set)

        self.model.annual_profit = pe.Constraint(rule=annual_profit)

        def hourly_profit(model, t):
            if model.market_data[t, 'tariff'] == 1:
                tariff_price = model.components_data["EP_HT_cost"]
            else:
                tariff_price = model.components_data["EP_LT_cost"]
            return model.C_profit_hourly[t] == 0.8 * tariff_price * model.P_grid_negative[t] * self.dt

        self.model.hourly_profit = pe.Constraint(self.model.time_set, rule=hourly_profit)

        print('Checkpoint 08: Profit constraints'
              ' successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_power_balance_equation(self):
        ev_ds_condition = None
        building_inclusion_condition = None
        if self.model_variant == 1:
            ev_ds_condition = 0
            building_inclusion_condition = 0
        elif self.model_variant == 2:
            ev_ds_condition = 1
            building_inclusion_condition = 0
        elif self.model_variant == 3:
            ev_ds_condition = 0
            building_inclusion_condition = 1
        elif self.model_variant == 4:
            ev_ds_condition = 1
            building_inclusion_condition = 1

        def power_balance_equation(model, t):
            return model.P_grid[t] + model.P_pv[t] + model.P_battery_ds[t] + \
                   sum(model.P_ev_ds[lot, t] for lot in model.parking_lot_set) * ev_ds_condition \
                   == \
                   sum(model.P_ev_ch[lot, t] for lot in model.parking_lot_set) + model.P_battery_ch[t] + \
                   model.P_OD_data[t] * building_inclusion_condition

        self.model.power_balance_equation = pe.Constraint(self.model.time_set, rule=power_balance_equation)

        def grid_power(model, t):
            return model.P_grid[t] == model.P_grid_positive[t] - model.P_grid_negative[t]

        self.model.grid_power = pe.Constraint(self.model.time_set, rule=grid_power)

        print('Checkpoint 09: Power balance equation'
              ' successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_grid_constraints(self):
        def monthly_peak_grid_power(model, t):
            return model.P_grid_max[int(model.market_data[t, 'month'])] >= \
                   model.P_grid_positive[t] + model.P_grid_negative[t]

        self.model.monthly_peak_grid_power = pe.Constraint(self.model.time_set, rule=monthly_peak_grid_power)

        def expected_peak_grid_power(model, t):
            return model.P_grid_max[t] <= model.P_contracted

        self.model.expected_peak_grid_power = pe.Constraint(self.model.month_set, rule=expected_peak_grid_power)

        if self.model_variant == 1 or self.model_variant == 2:
            def contracted_grid_power(model, t):
                return model.P_contracted == model.P_cs_contracted

            self.model.contracted_grid_power = pe.Constraint(self.model.month_set, rule=contracted_grid_power)
        else:
            def contracted_grid_power(model, t):
                return model.P_contracted == model.P_cs_contracted + model.components_data["BO_P_contracted_OD"]

            self.model.contracted_grid_power = pe.Constraint(self.model.month_set, rule=contracted_grid_power)

        def negative_grid_power(model, t):
            return model.P_grid_negative[t] <= model.P_pv[t]
        self.model.negative_grid_power = pe.Constraint(self.model.time_set, rule=negative_grid_power)

        print('Checkpoint 10: Grid constraints'
              ' successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_pv_system_constraints(self):
        def pv_system_power_production(model, t):
            return model.P_pv[t] == model.P_pv_install * model.pv_system_data[t]

        self.model.pv_system_power_production = pe.Constraint(self.model.time_set, rule=pv_system_power_production)

        def pv_system_optimal_install_power(model):
            return model.P_pv_install <= model.components_data["SP_max_power"] * model.binary_pv

        self.model.pv_system_optimal_install_power = pe.Constraint(rule=pv_system_optimal_install_power)

        print('Checkpoint 11: PV system constraints'
              ' successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_battery_system_constraints(self):
        def battery_charging_power(model, t):
            return model.P_battery_ch[t] <= model.P_battery_max[t]

        self.model.battery_charging_power = pe.Constraint(self.model.time_set, rule=battery_charging_power)

        def battery_discharging_power(model, t):
            return model.P_battery_ds[t] <= model.P_battery_MAX

        self.model.battery_discharging_power = pe.Constraint(self.model.time_set, rule=battery_discharging_power)

        def maximum_battery_capacity(model, t):
            return model.E_battery_capacity <= model.components_data["BS_max_capacity"] * model.binary_battery

        self.model.maximum_battery_capacity = pe.Constraint(self.model.time_set, rule=maximum_battery_capacity)

        def battery_capacity_lower_bound(model, t):
            return model.E_battery_capacity * model.components_data["BS_DoD"] <= model.E_battery[t]

        self.model.battery_capacity_lower_bound = pe.Constraint(self.model.time_set, rule=battery_capacity_lower_bound)

        def battery_capacity_upper_bound(model, t):
            return model.E_battery[t] <= model.E_battery_capacity

        self.model.battery_capacity_upper_bound = pe.Constraint(self.model.time_set, rule=battery_capacity_upper_bound)

        def battery_state_of_energy(model, t):
            if t == 1:
                return model.E_battery[t] == model.E_battery_capacity * model.components_data["BS_DoD"] + \
                       (model.P_battery_ch[t] * model.components_data["BS_charging_eff"]
                        - model.P_battery_ds[t] / model.components_data["BS_discharging_eff"]) * self.dt
            else:
                return model.E_battery[t] == model.E_battery[t - 1] + \
                       (model.P_battery_ch[t] * model.components_data["BS_charging_eff"]
                        - model.P_battery_ds[t] / model.components_data["BS_discharging_eff"]) * self.dt

        self.model.battery_state_of_energy = pe.Constraint(self.model.time_set, rule=battery_state_of_energy)

        def bat_ch_change1(model, t):
            if t == 1:
                return pe.Constraint.Skip
            else:
                return model.P_battery_ch[t] - model.P_battery_ch[t - 1] <= model.Bat_ch_ramp[t]

        self.model.bat_ch_change1 = pe.Constraint(self.model.time_set, rule=bat_ch_change1)

        def bat_ch_change2(model, t):
            if t == 1:
                return pe.Constraint.Skip
            else:
                return -(model.P_battery_ch[t] - model.P_battery_ch[t - 1]) <= model.Bat_ch_ramp[t]

        self.model.bat_ch_change2 = pe.Constraint(self.model.time_set, rule=bat_ch_change2)

        def bat_ds_change1(model, t):
            if t == 1:
                return pe.Constraint.Skip
            else:
                return model.P_battery_ds[t] - model.P_battery_ds[t - 1] <= model.Bat_ch_ramp[t]

        self.model.bat_ds_change1 = pe.Constraint(self.model.time_set, rule=bat_ds_change1)

        def bat_ds_change2(model, t):
            if t == 1:
                return pe.Constraint.Skip
            else:
                return -(model.P_battery_ds[t] - model.P_battery_ds[t - 1]) <= model.Bat_ch_ramp[t]

        self.model.bat_ds_change2 = pe.Constraint(self.model.time_set, rule=bat_ds_change2)

        def maximum_battery_power(model):
            return model.E_battery_capacity / 4 == model.P_battery_MAX

        self.model.maximum_battery_power = pe.Constraint(rule=maximum_battery_power)

        def constant_current_mode_battery_power(model, t):
            return model.P_battery_max[t] <= model.P_battery_MAX

        self.model.constant_current_mode_battery_power = pe.Constraint(self.model.time_set,
                                                                       rule=constant_current_mode_battery_power)

        def constant_voltage_mode_battery_power(model, t):
            return model.P_battery_max[t] <= (model.E_battery_capacity - model.E_battery[t]) / \
                   (4 - 4 * model.components_data["BS_CC_CV_switch"])

        self.model.constant_voltage_mode_battery_power = pe.Constraint(self.model.time_set,
                                                                       rule=constant_voltage_mode_battery_power)

        def constant_voltage_trend_battery_power(model, t):
            return model.P_battery_max_ch[t] == (model.E_battery_capacity - model.E_battery[t]) / \
                   (4 - 4 * model.components_data["BS_CC_CV_switch"])

        self.model.constant_voltage_trend_battery_power = pe.Constraint(self.model.time_set,
                                                                        rule=constant_voltage_trend_battery_power)

        print('Checkpoint 12: Battery system constraints'
              ' successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def create_electric_vehicle_constraints(self):
        def ev_state_of_energy_limits(model, lot, t):
            if model.PL_ev_available[t, lot] == 1:
                return model.E_ev[lot, t] <= model.E_ev_capacity[t, lot]
            else:
                return model.E_ev[lot, t] == 0

        self.model.ev_soe_limits = pe.Constraint(self.model.parking_lot_set, self.model.time_set,
                                                 rule=ev_state_of_energy_limits)

        if self.model_variant == 1 or self.model_variant == 3:
            def ev_state_of_energy(model, lot, t):
                if model.PL_ev_arrival[t, lot] == 1:
                    return model.E_ev[lot, t] == model.E_ev_capacity[t, lot] * model.EV_state_on_arrival[t, lot] \
                           + (model.P_ev_ch[lot, t] * model.components_data["PL_charging_eff"]) * self.dt
                elif model.PL_ev_available[t, lot] == 1:
                    return model.E_ev[lot, t] == model.E_ev[lot, t - 1] + \
                           + (model.P_ev_ch[lot, t] * model.components_data["PL_charging_eff"]) * self.dt
                else:
                    return pe.Constraint.Skip

            self.model.ev_soe = pe.Constraint(self.model.parking_lot_set, self.model.time_set, rule=ev_state_of_energy)
        elif self.model_variant == 2 or self.model_variant == 4:
            def ev_state_of_energy(model, lot, t):
                if model.PL_ev_arrival[t, lot] == 1:
                    return model.E_ev[lot, t] == model.E_ev_capacity[t, lot] * model.EV_state_on_arrival[t, lot] \
                           + (model.P_ev_ch[lot, t] * model.components_data["PL_charging_eff"]
                              - model.P_ev_ds[lot, t] / model.components_data["PL_discharging_eff"]) * self.dt
                elif model.PL_ev_available[t, lot] == 1:
                    return model.E_ev[lot, t] == model.E_ev[lot, t - 1] + \
                           + (model.P_ev_ch[lot, t] * model.components_data["PL_charging_eff"]
                              - model.P_ev_ds[lot, t] / model.components_data["PL_discharging_eff"]) * self.dt
                else:
                    return pe.Constraint.Skip

            self.model.ev_soe = pe.Constraint(self.model.parking_lot_set, self.model.time_set, rule=ev_state_of_energy)

            def ch_change3(model, lot, t):
                if model.PL_ev_arrival[t, lot] == 1:
                    return pe.Constraint.Skip
                elif model.PL_ev_available[t, lot] == 1:
                    return model.P_ev_ds[lot, t] - model.P_ev_ds[lot, t - 1] <= model.EV_ch_ramp[lot, t]
                else:
                    return pe.Constraint.Skip

            self.model.ch_change3 = pe.Constraint(self.model.parking_lot_set, self.model.time_set, rule=ch_change3)

            def ch_change4(model, lot, t):
                if model.PL_ev_arrival[t, lot] == 1:
                    return pe.Constraint.Skip
                elif model.PL_ev_available[t, lot] == 1:
                    return -(model.P_ev_ds[lot, t] - model.P_ev_ds[lot, t - 1]) <= model.EV_ch_ramp[lot, t]
                else:
                    return pe.Constraint.Skip

            self.model.ch_change4 = pe.Constraint(self.model.parking_lot_set, self.model.time_set, rule=ch_change4)

        def ev_relative_state_of_energy(model, lot, t):
            if model.PL_ev_available[t, lot] == 1:
                return model.SOE_ev_relative[lot, t] == model.E_ev[lot, t] / model.E_ev_capacity[t, lot]
            else:
                return model.SOE_ev_relative[lot, t] == 0

        self.model.ev_relative_soe = pe.Constraint(self.model.parking_lot_set, self.model.time_set,
                                                   rule=ev_relative_state_of_energy)

        def ch_change1(model, lot, t):
            if model.PL_ev_arrival[t, lot] == 1:
                return pe.Constraint.Skip
            elif model.PL_ev_available[t, lot] == 1:
                return model.P_ev_ch[lot, t] - model.P_ev_ch[lot, t - 1] <= model.EV_ch_ramp[lot, t]
            else:
                return pe.Constraint.Skip

        self.model.ch_change1 = pe.Constraint(self.model.parking_lot_set, self.model.time_set, rule=ch_change1)

        def ch_change2(model, lot, t):
            if model.PL_ev_arrival[t, lot] == 1:
                return pe.Constraint.Skip
            elif model.PL_ev_available[t, lot] == 1:
                return -(model.P_ev_ch[lot, t] - model.P_ev_ch[lot, t - 1]) <= model.EV_ch_ramp[lot, t]
            else:
                return pe.Constraint.Skip

        self.model.ch_change2 = pe.Constraint(self.model.parking_lot_set, self.model.time_set, rule=ch_change2)

        if self.end_soe_equal_to_requested_soe:
            upper_bound = 1
            lower_bound = 1
        else:
            upper_bound = 1.05
            lower_bound = 0.95

        def ev_relative_end_state_of_energy_upper_bound(model, lot, t):
            if model.PL_ev_departure[t, lot] == 1:
                return model.SOE_ev_relative[lot, t] <= model.EV_required_end_state[t, lot] * upper_bound
            else:
                return pe.Constraint.Skip

        self.model.ev_relative_end_soe_upper_bound = pe.Constraint(self.model.parking_lot_set, self.model.time_set,
                                                                   rule=ev_relative_end_state_of_energy_upper_bound)

        def ev_relative_end_state_of_energy_lower_bound(model, lot, t):
            if model.PL_ev_departure[t, lot] == 1:
                return model.SOE_ev_relative[lot, t] >= model.EV_required_end_state[t, lot] * lower_bound
            else:
                return pe.Constraint.Skip

        self.model.ev_relative_end_soe_lower_bound = pe.Constraint(self.model.parking_lot_set, self.model.time_set,
                                                                   rule=ev_relative_end_state_of_energy_lower_bound)

        def constant_current_mode_ev_charging_power(model, lot, t):
            if model.PL_ev_available[t, lot] == 1:
                return model.P_ev_ch[lot, t] <= 22
            else:
                return pe.Constraint.Skip

        self.model.constant_current_mode_ev_charging_power = pe.Constraint(self.model.parking_lot_set,
                                                                           self.model.time_set,
                                                                           rule=constant_current_mode_ev_charging_power)

        def constant_voltage_mode_ev_charging_power(model, lot, t):
            if model.PL_ev_available[t, lot] == 1:
                return model.P_ev_ch[lot, t] <= 22 * (1 - model.SOE_ev_relative[lot, t]) \
                       / (1 - model.components_data["BS_CC_CV_switch"])
            else:
                return model.P_ev_ch[lot, t] == 0

        self.model.constant_voltage_mode_ev_charging_power = pe.Constraint(self.model.parking_lot_set,
                                                                           self.model.time_set,
                                                                           rule=constant_voltage_mode_ev_charging_power)

        if self.model_variant == 2 or self.model_variant == 4:
            def ev_discharging_power(model, lot, t):
                if model.PL_ev_available[t, lot] == 1:
                    return model.P_ev_ds[lot, t] <= 22
                else:
                    return model.P_ev_ds[lot, t] == 0

            self.model.ev_discharging_power = pe.Constraint(self.model.parking_lot_set, self.model.time_set,
                                                            rule=ev_discharging_power)

        print('Checkpoint 13: Electric vehicle constraints'
              ' successfully created.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def solve_model(self):
        print('\n')
        print('Checkpoint 16: Preparing & solving the model...', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        solver = pyomo.opt.SolverFactory('gurobi')
        # tee=True -> see optimisation results in console; tee=False -> don't print the results in the console
        results = solver.solve(self.model, tee=True, keepfiles=False, options_string="mipgap=0.01 Method=2 MIPFocus=3")
        # print(results)

        if results.solver.status == pyomo.opt.SolverStatus.ok:
            logging.info('Solver status: OK.')
        else:
            logging.warning('Solver status: WARNING - Not OK!')
        if results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal:
            logging.info('Solver optimisation status: Optimal.')
            print('Checkpoint 17: Model solved.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        else:
            logging.warning('Solver optimisation status: WARNING - Not optimal!')
        self.model.solutions.load_from(results)

    def extract_results(self):
        self.model.results = dict()
        self.model.results['charging_schedule'] = []
        self.model.results['year_data'] = {}
        self.model.results['month_data'] = {}
        self.model.results['quarter_data'] = {}
        self.model.results['variables'] = {}

        self.model.results_dataframe = dict()
        self.model.results_dataframe['single_year_data'] = {}
        self.model.results_dataframe['per_month_data'] = {}
        self.model.results_dataframe['per_quarter_data'] = {}
        self.model.results_dataframe['optimal_variables'] = {}

        for variable in self.model.component_objects(Var, active=True):
            data_rows = len(variable)

            if variable.dim() == 2:
                results_data_2 = dict()

                for index in variable:
                    results_data_2[index[0]] = {}

                for index in variable:
                    results_data_2[index[0]][index[1]] = variable[index].value

                self.model.results['charging_schedule'].append({
                    'schedule': pd.DataFrame.from_dict(results_data_2, orient='index'),
                    'variable_name': str(variable)
                })

            temp_results = []
            for index in variable:
                temp_results.append(variable[index].value)

            if data_rows == 35040:
                self.model.results['year_data'][str(variable)] = temp_results
            if data_rows == 12:
                self.model.results['month_data'][str(variable)] = temp_results
            if data_rows == 4:
                self.model.results['quarter_data'][str(variable)] = temp_results
            if data_rows == 1:
                self.model.results['variables'][str(variable)] = temp_results

        self.model.results_dataframe['single_year_data'] = pd.DataFrame.from_dict(self.model.results['year_data'])
        self.model.results_dataframe['per_month_data'] = pd.DataFrame.from_dict(self.model.results['month_data'])
        self.model.results_dataframe['per_quarter_data'] = pd.DataFrame.from_dict(self.model.results['quarter_data'])
        self.model.results_dataframe['optimal_variables'] = pd.DataFrame.from_dict(self.model.results['variables'])

        print('Results extracted from solved optimised model.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def save_optimisation_results_to_excel(self):
        directory_path = os.path.dirname(os.path.realpath(__file__))

        main_directory = '\\optimisation_results\\'

        if self.end_soe_equal_to_requested_soe:
            file_directory = 'results_required\\'
        else:
            file_directory = 'results\\'

        file_subdirectory = f'model{self.model_variant}\\'

        full_dir_path = directory_path + main_directory + file_directory + file_subdirectory
        if not os.path.exists(full_dir_path):
            os.makedirs(full_dir_path)

        file_name = 'optimal_ev_charging_schedule.xlsx'
        # file_writer = pd.ExcelWriter(directory_path + file_directory + file_subdirectory + file_name)
        file_writer = pd.ExcelWriter(full_dir_path + file_name)
        for item in self.model.results['charging_schedule']:
            item['schedule'].T.to_excel(file_writer, item['variable_name'])
        file_writer.save()
        print('Saved optimal charging schedule results.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        file_name = 'nanogrid_optimisation_results.xlsx'
        # file_writer = pd.ExcelWriter(directory_path + file_directory + file_subdirectory + file_name)
        file_writer = pd.ExcelWriter(full_dir_path + file_name)
        self.model.results_dataframe['single_year_data'].to_excel(file_writer, '15-min optimal data for 1 year')
        self.model.results_dataframe['optimal_variables'].to_excel(file_writer, 'Optimal variables')
        self.model.results_dataframe['per_quarter_data'].to_excel(file_writer, 'Optimal quarterly variables')
        self.model.results_dataframe['per_month_data'].to_excel(file_writer, 'Optimal monthly variables')
        file_writer.save()
        print('Saved optimal variables and data values.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def load_data():
    directory_path = os.path.dirname(os.path.realpath(__file__))

    temp_data = dict()

    xls_name = "\\data\\components_data.csv"
    components_data = pd.read_csv(directory_path + xls_name,
                                  sep=";", header=0, index_col=[1], decimal=',')
    temp_data['components'] = components_data

    xls_name = "\\data\\parking_lot_data.csv"
    parking_lot_data = pd.read_csv(directory_path + xls_name,
                                   sep=";", header=[0, 1], index_col=[0], skiprows=1, decimal='.')
    temp_data['parking_lot'] = parking_lot_data

    xls_name = "\\data\\market_data.csv"
    market_data = pd.read_csv(directory_path + xls_name,
                              sep=";", header=[0], index_col=[0], skiprows=1, decimal=',')
    temp_data['market'] = market_data

    xls_name = "\\data\\pv_data.csv"
    pv_data = pd.read_csv(directory_path + xls_name,
                          sep=";", header=[0], index_col=[0], skiprows=1, parse_dates=True, decimal=',')
    temp_data['pv_system'] = pv_data

    xls_name = "\\data\\building_data_2016.csv"
    building_data = pd.read_csv(directory_path + xls_name,
                                sep=";", header=[0], index_col=[0], skiprows=1, parse_dates=True, decimal=',')
    temp_data['building'] = building_data

    print('Checkpoint 00: All data loaded successfully.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return temp_data


if __name__ == '__main__':
    print('Program started')
    print('\nLoading data for optimisation process...', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    data = load_data()

    models = [1, 2, 3, 4]
    # models = [1, 2]
    # models = [4]
    # If = True -> charge electric vehicles to the requested state of energy (soe)
    # If = False -> end soe can have values in range of +/-5% of the requested soe
    # end_soe_to_equal_requested_soe = True
    end_soe_to_equal_requested_soe = False

    for model_scheme in models:
        print('\n')
        print('------------------------MODEL {}------------------------'.format(model_scheme))
        print('Constructing nanogrid optimisation model with the {} model scheme\n'.format(model_scheme))
        nanogrid = Nanogrid(
            data['components'], data['parking_lot'], data['market'], data['pv_system'], data['building'],
            model_scheme, end_soe_to_equal_requested_soe
        )
        nanogrid.solve_model()

        nanogrid.extract_results()
        nanogrid.save_optimisation_results_to_excel()
        del nanogrid

    print('\n')
    print('Program finished.', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
