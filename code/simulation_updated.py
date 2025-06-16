import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa import Model
import random
from scipy.optimize import root_scalar

random.seed(0)

step_durtion = 5

# HFCV utility
beta_nu = 1  
epsilon_t = 10

# EV utility
alpha = 1
Pr = 0.042
sigma = 5 / 60 * step_durtion


class EV:
    def __init__(self, capacity, initial_charge, charge_speed):
        self.charging = False
        self.capacity = capacity
        self.remaining_charge = initial_charge
        self.charge_speed = charge_speed
        self.waiting_time = 0

    def balking(self, quque_length):
        if (self.remaining_charge / self.capacity >= 0.5 and quque_length >= 6):
            return False
        else:
            return True
    
    def start_charging(self, price, charge_level, time_factor):
        self.charging = True
        self.charge_amount = self.utility_function(price, charge_level, time_factor)
    
    def utility_function(self, price, charge_level, time_factor):
        total_charging_amount = 0
        total_budget = 0.1 * charge_level
        time_penalty = self.waiting_time
        tmp_step = 0
        while (tmp_step * step_durtion <= (self.capacity - self.remaining_charge) / self.charge_speed):
            tmp_step += 1
            U_pur = alpha * np.log10(step_durtion * self.charge_speed + 1)
            C_ELE = step_durtion * self.charge_speed * price
            y = (-(U_pur * Pr) - C_ELE / 1 - sigma * time_penalty * time_factor) * 1
            time_penalty = step_durtion
            total_budget += y
            total_charging_amount += step_durtion * self.charge_speed
            if (total_charging_amount > self.capacity - self.remaining_charge):
                total_charging_amount = self.capacity - self.remaining_charge
                break
            if (total_budget <= 0):
                break
        return total_charging_amount



class HFCV:
    def __init__(self, capacity, initial_charge):
        self.refueling = False
        self.capacity = capacity
        self.remaining_charge = capacity - initial_charge
    
    def start_refueling(self, price, charge_level, available_stations):
        if (available_stations > 0):
            self.refueling = True
            self.refuel_amount = self.utility_function(price, charge_level)
        else:
            self.refueling = False
            self.refuel_amount = 0
            
    def compute_h_star(self, pi_t, H_nu):
        numerator = -beta_nu * epsilon_t * np.exp(beta_nu)
        denominator = pi_t * (1 - np.exp(beta_nu))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = numerator / denominator
            h_star = (H_nu / beta_nu) * np.log(ratio)
        return h_star
    
    def utility_function(self, price, charge_level):
        return self.compute_h_star(price, charge_level)


class ChargingStation(Model):
    def __init__(self, E_spot, H_spot, lambda_ev, lambda_hfcv, hfcv_refuel_time, price_rate = 1, wait_threshold = 30, time_factor = 1,
                 solar_efficiency=0.2, wind_efficiency=0.3, elec_to_hydrogen_rate=0.7, hydrogen_to_elec_rate=0.6,
                 elec_storage_capacity=500, hydrogen_storage_capacity=300):
        super().__init__()
        self.num_hydrogen_spots = H_spot
        self.num_electric_spots = E_spot
        self.time_hydrogen_spots = []
        self.time_electric_spots = []
        self.speed_electric_spots = []
        for _ in range(self.num_hydrogen_spots):
            self.time_hydrogen_spots.append(0)
        for _ in range(self.num_electric_spots):
            self.time_electric_spots.append(0)
            self.speed_electric_spots.append(0)
        self.ev_queue = []
        self.hfcv_queue = []
        self.hfcv_refuel_time = hfcv_refuel_time
        self.total_profit = 0

        # Solar power related
        self.tau = 0.9
        self.eta = 0.15
        self.gamma = 0.0045
        self.Area = 300

        # Wind power related
        self.V_ci = 3    # Cut-in speed (mph)
        self.V_co = 25   # Cut-out speed (mph)
        self.V_r = 13    # Rated wind turbine speed (mph)
        self.R_W = 500   # Rated power level (kW)

        # Electrolyzer related
        self.eta_Z = 0.6  # Electrolyzer efficiency
        self.kappa_ph = 1 / 55  # Conversion factor (kg/kWh)
        self.R_Z = 1200  # Rated capacity (kW)
        self.alpha_Z = 0.05  # Lower limit coefficient
        self.kappa_hh = 2 # Required energy for compressing one unit hydrogen 

        # Fuel cell related
        self.eta_F = 0.65  # Fuel cell efficiency
        self.kappa_hp = 35  # Conversion factor (kWh/kg)
        self.R_H = 50  # Rated flow capacity (kg/h)
        self.alpha_H = 0.05  # Lower limit coefficient
        
        # Storage capacities
        self.electricity_storage = 0
        self.hydrogen_storage = 0
        self.elec_storage_capacity = elec_storage_capacity
        self.hydrogen_storage_capacity = hydrogen_storage_capacity
        
        # Efficiency and conversion rates
        self.solar_efficiency = solar_efficiency
        self.wind_efficiency = wind_efficiency
        self.elec_to_hydrogen_rate = elec_to_hydrogen_rate
        self.hydrogen_to_elec_rate = hydrogen_to_elec_rate
        
        # Poisson arrival rates
        self.lambda_ev = lambda_ev / 60 * step_durtion
        self.lambda_hfcv = lambda_hfcv/ 60 * step_durtion

        # Price rate
        self.price_rate = price_rate
        
        # queue
        self.EV_queue = 0

        self.EV_wait = []
        self.wait_threshold = wait_threshold
        self.time_factor = time_factor

        self.data = {
            "EVs Arrived": [],
            "HFCVs Arrived": [],
            "EVs Served": [],
            "HFCVs Served": [],
            "EVs Fully Charged": [],
            "HFCVs Fully Charged": [],
            "Total EVs Charged Amount": [],
            "Total HFCVs Charged Amount": [],
            "Charging Station Profit": [],
            "Electricity Storage": [],
            "Hydrogen Storage": [],
            "Electricity Spot": [],
            # "Hydrogen Spot": [],
            "EVs queue": [],
            # "HFCVs queue": []
            "Arrived EVs Total Wait Time": [],
            "Served EVs Total Charging Time": [],
            "Generated E": [],
            "Transferred E": [],
            "Transferred H": [],
            "Electricity Profit": [],
            "Hydrogen Profit": [],
        }

        self.load_data()

        self.charge_stat = []

    def load_data(self):
        file_path = '5-Minute_Expanded_Data.xlsx'
        df_xlsx_s1 = pd.read_excel(file_path, sheet_name='Sheet1')

        Pp_t_values = df_xlsx_s1.iloc[:, 0].values # external electricity purchasing price ($/kw)
        Ps_t_values = df_xlsx_s1.iloc[:, 1].values # external electricity selling price ($/kw)
        Omega_values = df_xlsx_s1.iloc[:, 2].values # this is the solar irradiation data (KW/m2)
        V_values = df_xlsx_s1.iloc[:, 3].values  # this is wind speed data (m/s)
        Phi_values = df_xlsx_s1.iloc[:, 4].values # this is the ambient temperature data (C)

        # Demand_BU_values = df_xlsx_s1.iloc[:, 6:9].values # building power demand (kw)
        # Demand_H2_values = df_xlsx_s1.iloc[:, 9].values # hydrogen demand (kg)
        HFCV_arrivals_values = df_xlsx_s1.iloc[:, 5].values # hydrogen vehuicle arrivals
        Epsilon_value = df_xlsx_s1.iloc[:, 6].values # epsilon value (expected hydrogen price)
        EV_arrivals_input = df_xlsx_s1.iloc[:,7].values # ev arrival number

        # beta_rand_value = df_xlsx_s2.iloc[:, 1].values # random beta value between 0.8-3
        # H_rand_value = df_xlsx_s2.iloc[:, 2].values # random hydrogen demand H value between 6*0.4-6*0.98

        # Size_EVs_input = df_xlsx_s3.iloc[:, 1].values  # battery size for swapped battery
        # Ini_EVs_input = df_xlsx_s3.iloc[:, 2].values  # initial battery soc level for ev


        self.omega = list(Omega_values)
        self.phi = list(Phi_values)
        self.wind_speed = list(V_values)
        self.Pp = list(Pp_t_values)
        self.Ps = list(Ps_t_values)

        self.H_price = list(Epsilon_value)
        self.EV_num = list(EV_arrivals_input)
        self.HFCV_num = list(HFCV_arrivals_values)
    
    def generate_solar_energy(self, t):
        generated = self.omega[t] * self.tau * self.eta * self.Area * (1 - self.gamma * (self.phi[t] - 25)) / 60 * step_durtion
        self.store_electricity(generated)
        return generated
    
    def generate_wind_energy(self, t):
        if self.V_r <= self.wind_speed[t] <= self.V_co:
            generated = self.R_W
        elif self.V_ci <= self.wind_speed[t] < self.V_r:
            generated = self.R_W * ((self.wind_speed[t]**3 - self.V_ci**3) / (self.V_r**3 - self.V_ci**3))
        else:
            generated = 0  

        generated = generated / 60 * step_durtion

        self.store_electricity(generated)
        return generated

    
    def electricity_to_hydrogen(self, electricity):

        # Constrain input power within operational limits
        min_power = self.alpha_Z * self.R_Z
        if (min_power > electricity):
            return 0
        electricity = max(min_power, min(electricity, self.R_Z))

        # Compute hydrogen production based on the formula
        hydrogen_produced = self.eta_Z * electricity * self.kappa_ph

        # Store the produced hydrogen and update electricity storage
        self.store_hydrogen(hydrogen_produced)

        # compress all hydrogen
        pC = self.kappa_hh * hydrogen_produced
        self.electricity_storage = self.electricity_storage - electricity - pC
        return hydrogen_produced


    def hydrogen_to_electricity(self, hydrogen):
        # Constrain hydrogen input within operational limits
        min_hydrogen = self.alpha_H * self.R_H
        if (min_hydrogen > hydrogen):
            return 0
        hydrogen = max(min_hydrogen, min(hydrogen, self.R_H))

        # Compute electricity production based on the formula
        electricity_produced = self.eta_F * hydrogen * self.kappa_hp

        # Store the generated electricity and update hydrogen storage
        self.store_electricity(electricity_produced)
        self.hydrogen_storage -= hydrogen  # Reduce storage based on used hydrogen

        return electricity_produced
    
    def purchase_electricity(self, amount, t):
        self.total_profit -= min(self.elec_storage_capacity - self.electricity_storage, amount) * self.Pp[t] * self.price_rate
        self.electricity_storage = min(self.electricity_storage + amount, self.elec_storage_capacity)
    
    def purchase_hydrogen(self, amount, t):
        self.total_profit -= min(self.hydrogen_storage_capacity - self.hydrogen_storage, amount) * self.H_price[t] * self.price_rate
        self.hydrogen_storage = min(self.hydrogen_storage + amount, self.hydrogen_storage_capacity)
    
    def sell_electricity(self, amount, t):
        self.total_profit += min(self.electricity_storage, amount) * self.Ps[t] * self.price_rate
        self.electricity_storage = max(self.electricity_storage - amount, 0)

    def store_electricity(self, amount):
        self.electricity_storage = min(self.electricity_storage + amount, self.elec_storage_capacity)
    
    def store_hydrogen(self, amount):
        self.hydrogen_storage = min(self.hydrogen_storage + amount, self.hydrogen_storage_capacity)
    
    def step(self, tmp_num):
        tmp_E_price = (self.Pp[tmp_num] + (self.Pp[tmp_num] - self.Ps[tmp_num]) / 2) * self.price_rate
        # print(tmp_E_price)
        # 3 charging speed levels
        charge_speed = [8, 10, 12]
        EV_capacity = [40, 80, 120]
        # Simulating energy generation
        E_s = self.generate_solar_energy(tmp_num)
        E_w = self.generate_wind_energy(tmp_num)
        # print(tmp_num, self.electricity_storage, self.hydrogen_storage)
        
        # Simulating vehicle arrivals
        num_new_evs = np.random.poisson(self.lambda_ev)
        num_new_hfcvs = np.random.poisson(self.lambda_hfcv)
        
        # arrival
        self.data["EVs Arrived"].append(num_new_evs)
        self.data["HFCVs Arrived"].append(num_new_hfcvs)

        # transferring between E and H
        E_H = 0
        H_E = 0
        if (num_new_evs * tmp_E_price * 80 >= num_new_hfcvs * self.H_price[tmp_num] * 6):
            H_E = self.electricity_to_hydrogen(self.electricity_storage)
        else:
            E_H = self.hydrogen_to_electricity(self.hydrogen_storage)
        self.data["Generated E"].append(E_s+E_w)
        self.data["Transferred E"].append(E_H)
        self.data["Transferred H"].append(H_E)
        if (self.electricity_storage < 0):
            self.purchase_electricity(-self.electricity_storage, tmp_num)
        if (self.hydrogen_storage < 0):
            self.purchase_hydrogen(-self.hydrogen_storage, tmp_num)

        # queue
        for i in range(num_new_evs):
            type = random.choice([0, 1, 2])
            ev = EV(EV_capacity[type], np.random.uniform(EV_capacity[type]/10, EV_capacity[type]/2), charge_speed[type])
            if (ev.balking(self.EV_queue)):
                self.EV_queue += 1
                self.EV_wait.append(ev)

        # charge
        charging_num = min(self.EV_queue, self.num_electric_spots)
        refueling_num = min(num_new_hfcvs, self.num_hydrogen_spots)

        self.data["EVs Fully Charged"].append(0)
        self.data["HFCVs Fully Charged"].append(0)
        self.data["Arrived EVs Total Wait Time"].append(0)
        self.data["Served EVs Total Charging Time"].append(0)
        self.data["Total EVs Charged Amount"].append(0)
        self.data["Total HFCVs Charged Amount"].append(0)
        charging_count = 0
        refueling_count = 0
        served_EV = 0
        served_HFCV = 0
        for i in range(charging_num):
            self.EV_wait[0].start_charging(price=tmp_E_price, charge_level=self.EV_wait[0].capacity-self.EV_wait[0].remaining_charge, time_factor = self.time_factor)
            if self.EV_wait[0].charging:
                self.data["Arrived EVs Total Wait Time"][tmp_num] += self.EV_wait[0].waiting_time
                # find the empty spot
                tmp_id = -1
                for j in range(len(self.time_electric_spots)):
                    if (self.time_electric_spots[j] == 0):
                        tmp_id = j
                        break
                self.num_electric_spots -= 1
                self.charge_stat.append([self.EV_wait[0].capacity, self.EV_wait[0].capacity-self.EV_wait[0].remaining_charge, self.EV_wait[0].charge_amount, self.EV_wait[0].waiting_time, tmp_E_price])
                if (self.EV_wait[0].charge_amount == self.EV_wait[0].capacity-self.EV_wait[0].remaining_charge):
                    self.data["EVs Fully Charged"][-1] += 1
                self.time_electric_spots[tmp_id] = self.EV_wait[0].charge_amount / self.EV_wait[0].charge_speed
                self.data["Served EVs Total Charging Time"][tmp_num] += self.EV_wait[0].charge_amount / self.EV_wait[0].charge_speed * step_durtion
                # print(self.EV_wait[0].charge_amount, self.EV_wait[0].charge_amount / self.EV_wait[0].charge_speed)
                self.data["Total EVs Charged Amount"][tmp_num] += self.EV_wait[0].charge_amount
                self.speed_electric_spots[tmp_id] = self.EV_wait[0].charge_speed
                served_EV += 1
                self.EV_queue -= 1
                del(self.EV_wait[0])

        tmp_H = self.num_hydrogen_spots
        # HFCV refuel
        for i in range(refueling_num):
            hfcv = HFCV(capacity=6, initial_charge=np.random.uniform(1, 6))
            hfcv.start_refueling(self.H_price[tmp_num]*self.price_rate, 6-hfcv.remaining_charge, self.num_hydrogen_spots)
            if hfcv.refueling:
                # print(hfcv.refuel_amount)
                if (self.hydrogen_storage >= hfcv.refuel_amount):
                    refueling_count += hfcv.refuel_amount
                    self.hydrogen_storage -= hfcv.refuel_amount
                else:
                    # assume the profit is half
                    refueling_count += (self.hydrogen_storage + (min(hfcv.refuel_amount, 6-hfcv.remaining_charge)-self.hydrogen_storage)/2)
                    self.hydrogen_storage = 0
                if (hfcv.refuel_amount == 6-hfcv.remaining_charge):
                    self.data["HFCVs Fully Charged"][-1] += 1
                served_HFCV += 1
                self.num_hydrogen_spots -= 1
                self.data["Total HFCVs Charged Amount"][tmp_num] += hfcv.refuel_amount
        self.num_hydrogen_spots = tmp_H
        
        self.data["Electricity Spot"].append(self.num_electric_spots)
        # next time episode
        for j in range(len(self.time_electric_spots)):
            if (self.time_electric_spots[j] > 0):
                if (self.electricity_storage >= self.speed_electric_spots[j]):
                    charging_count += self.speed_electric_spots[j]
                    self.electricity_storage -= self.speed_electric_spots[j]
                else:
                    charging_count += self.electricity_storage
                    self.electricity_storage = 0
                if (self.time_electric_spots[j] <= 1):
                    self.num_electric_spots += 1
                    self.time_electric_spots[j] = 0
                else:
                    self.time_electric_spots[j] -= 1
        
        profit_E = (charging_count * tmp_E_price) * self.price_rate
        profit_H = (refueling_count * self.H_price[tmp_num]) * self.price_rate
        profit = profit_E + profit_H
        if (len(self.data["Charging Station Profit"]) > 0):
            profit = profit + self.data["Charging Station Profit"][-1]
            self.data["Electricity Profit"].append(self.data["Electricity Profit"][-1] + profit_E)
            self.data["Hydrogen Profit"].append(self.data["Hydrogen Profit"][-1] + profit_H)
        else:
            self.data["Electricity Profit"].append(profit_E)
            self.data["Hydrogen Profit"].append(profit_H)

        # update waiting time and EV leaving
        for i in range(len(self.EV_wait) - 1, -1, -1):
            self.EV_wait[i].waiting_time += step_durtion
            if self.EV_wait[i].waiting_time >= self.wait_threshold:
                self.data["Arrived EVs Total Wait Time"][tmp_num] += self.wait_threshold
                self.EV_queue -= 1
                del self.EV_wait[i]
        self.data["Charging Station Profit"].append(profit)
        

        ########## think about the time for queue length and spot number
        self.data["EVs queue"].append(self.EV_queue)
        
        self.data["EVs Served"].append(served_EV)
        self.data["HFCVs Served"].append(served_HFCV)
        # Store energy states
        self.data["Electricity Storage"].append(self.electricity_storage)
        self.data["Hydrogen Storage"].append(self.hydrogen_storage)


# Run the simulation
model = ChargingStation(4, 2, lambda_ev=8, lambda_hfcv=8, hfcv_refuel_time=step_durtion, price_rate=1)

for i in range(288):
    model.step(i)


columns = ['Capacity', 'Total Demand', 'Charge_Amount', 'Waiting_Time', 'Electricity_Price']
df = pd.DataFrame(model.charge_stat, columns=columns)
df.to_excel('charge_stats.xlsx', index=False)


df = pd.DataFrame(model.data)
df.to_excel('charging_station_simulation_results.xlsx', index=False)


# Plot EV Arrivals and Served
# Assuming df is already defined
time_steps = df.index

# Generate time labels starting from 00:00 with 5-minute intervals
time_labels = pd.date_range(start='00:00', periods=len(time_steps), freq='5min').strftime('%H:%M')

arrived = df['EVs Arrived'].values
served = df['EVs Served'].values

# plt.figure(figsize=(14, 6))
# bar_width = 0.8

# # Plot Arrived
# plt.bar(time_labels, arrived, width=bar_width, color='skyblue', label='EVs Arrived')

# # Plot Served on top
# plt.bar(time_labels, served, width=bar_width, color='orange', label='EVs Served')

# plt.xlabel("Time of Day")
# plt.ylabel("EV Count")
# plt.title("EVs Arrived and Served Over Time")
# plt.xticks(ticks=np.arange(0, len(time_labels), 24), labels=time_labels[::24], rotation=45)  # Show every 2 hours
# plt.legend()
# plt.tight_layout()
# plt.show()



# # Create time labels for x-axis
# time_labels = pd.date_range(start="00:00", periods=len(df), freq="5min").strftime('%H:%M')

# # Create the figure and axes
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # Primary y-axis: Electricity Storage
# color1 = 'tab:orange'
# ax1.set_xlabel('Time of Day')
# ax1.set_ylabel('Electricity Storage (kWh)', color=color1)
# line1, = ax1.plot(time_labels, df['Electricity Storage'], color=color1, label='Electricity Storage')
# ax1.tick_params(axis='y', labelcolor=color1)
# ax1.set_xticks(range(0, len(time_labels), 24))  # one tick every 2 hours
# ax1.set_xticklabels(time_labels[::24], rotation=45)

# # Secondary y-axis: Hydrogen Storage
# ax2 = ax1.twinx()
# color2 = 'tab:blue'
# ax2.set_ylabel('Hydrogen Storage (kg)', color=color2)
# line2, = ax2.plot(time_labels, df['Hydrogen Storage'], color=color2, label='Hydrogen Storage')
# ax2.tick_params(axis='y', labelcolor=color2)

# # Title and legend
# fig.suptitle('Electricity and Hydrogen Storage Levels Over Time', fontsize=14)
# ax1.legend(handles=[line1, line2], loc='upper right')

# # Final touches
# ax1.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()



# # Assume df contains 'Generated E', 'Transferred E', 'Transferred H'
# # Create time labels for x-axis
# time_labels = pd.date_range(start="00:00", periods=len(df), freq="5min").strftime('%H:%M')
# x = np.arange(len(df))

# # Initialize figure and axis
# fig, ax1 = plt.subplots(figsize=(14, 6))

# # Plot top bars (Electricity)
# bar1 = ax1.bar(x - 0.2, df["Generated E"], width=0.4, color='tab:green', label='Generated Electricity')
# bar2 = ax1.bar(x + 0.2, df["Transferred E"], width=0.4, color='tab:orange', label='Transferred to Hydrogen')

# # Formatting for electricity axis
# ax1.set_ylabel("Electricity (kWh)")
# ax1.set_xticks(np.arange(0, len(x), 24))  # one tick every 2 hours
# ax1.set_xticklabels(time_labels[::24], rotation=45)
# ax1.set_xlabel("Time of Day")
# ax1.set_ylim(0, max(df["Generated E"].max(), df["Transferred E"].max()) * 1.2)

# # Create secondary axis for hydrogen bars
# ax2 = ax1.twinx()

# # Plot bottom bars (Hydrogen), inverted to go below x-axis
# bar3 = ax2.bar(x, -df["Transferred H"], width=0.4, color='tab:blue', label='Transferred to Hydrogen Storage')

# # Formatting for hydrogen axis
# ax2.set_ylabel("Hydrogen (kg)")
# ax2.set_ylim(-df["Transferred H"].max() * 1.2, 0)
# ax2.tick_params(axis='y', labelcolor='tab:blue')

# # Title and legend
# plt.title("Generated and Transferred Energy Over Time")
# bars = [bar1, bar2, bar3]
# labels = [bar.get_label() for bar in bars]
# ax1.legend(bars, labels, loc="upper right")

# # Draw horizontal axis at y=0
# ax1.axhline(0, color='black', linewidth=0.8)

# plt.tight_layout()
# plt.show()



# # Assume df["Generated E"] is the original 5-minute data (length = 288)
# # Aggregate to hourly (12 time steps per hour)
# df_hourly = df["Generated E"].groupby(np.arange(len(df)) // 12).sum()

# # Create hourly time labels
# time_labels = pd.date_range(start="00:00", periods=24, freq="1H").strftime('%H:%M')

# # Plot
# plt.figure(figsize=(12, 5))
# plt.bar(time_labels, df_hourly, color='tab:green', width=0.8)
# plt.title("Hourly Generated Electricity")
# plt.xlabel("Time of Day")
# plt.ylabel("Generated Electricity (kWh)")
# plt.xticks(rotation=45)
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()



# # Generate time labels (assuming 5-minute intervals)
# time_labels = pd.date_range(start='00:00', periods=len(df), freq='5min').strftime('%H:%M')

# # Prepare data
# x = range(len(df))
# electricity_spots = df['Electricity Spot']
# ev_queue = df['EVs queue']

# # Plot
# plt.figure(figsize=(14, 6))

# # Bars for electricity spots (above x-axis)
# plt.bar(x, electricity_spots, color='tab:blue', label='Available Electricity Spot')

# # Bars for EV queue (below x-axis, offset vertically)
# plt.bar(x, ev_queue, bottom=[-val for val in ev_queue], color='tab:red', label='EV Queue Length')

# # Horizontal axis at y=0
# plt.axhline(0, color='black', linewidth=0.8)

# # Format ticks
# plt.xticks(
#     ticks=range(0, len(df), 12),  # every hour (12 intervals of 5 mins)
#     labels=time_labels[::12],
#     rotation=45
# )

# plt.title("Charging Spot Availability and EV Queue Length Over Time")
# plt.xlabel("Time of Day")
# plt.ylabel("Count")
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.grid(axis='y', linestyle='--', alpha=0.4)
# plt.show()




# # Time axis in HH:MM (assuming each step = 5 mins)
# time_labels = pd.date_range(start='00:00', periods=len(df), freq='5min').strftime('%H:%M')
# x = range(len(df))

# arrived = df["HFCVs Arrived"]
# served = df["HFCVs Served"]

# plt.figure(figsize=(14, 5))

# # Plot the total arrivals in blue
# plt.bar(x, arrived, width=0.8, label='HFCVs Arrived', color='tab:blue')

# # Plot the served portion in green (bottom-aligned with 0)
# plt.bar(x, served, width=0.8, label='HFCVs Served', color='tab:green')

# # Time x-axis formatting
# plt.xticks(
#     ticks=range(0, len(df), 12),  # every hour
#     labels=time_labels[::12],
#     rotation=45
# )

# plt.title("HFCV Arrivals and Served per Time Step")
# plt.xlabel("Time of Day")
# plt.ylabel("Count")
# plt.legend(loc='upper right')
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()


# # # Plot Charging Station Profit
# # Time axis as HH:MM
# time_labels = pd.date_range(start='00:00', periods=len(df), freq='5min').strftime('%H:%M')
# x = range(len(df))

# # Extract the data
# total_profit = df["Charging Station Profit"]
# elec_profit = df["Electricity Profit"]
# hydro_profit = df["Hydrogen Profit"]
# # print(hydro_profit)
# # Plotting
# plt.figure(figsize=(14, 6))

# # Stacked bars for Electricity and Hydrogen profits
# plt.bar(x, elec_profit, label='Electricity Profit', color='tab:blue')
# plt.bar(x, hydro_profit, bottom=elec_profit, label='Hydrogen Profit', color='tab:green')

# # Line plot for total profit
# plt.plot(x, total_profit, label='Total Profit', color='black', linewidth=2)

# # Time ticks
# plt.xticks(
#     ticks=range(0, len(df), 12),
#     labels=time_labels[::12],
#     rotation=45
# )

# plt.title("Cumulative Profit Over Time")
# plt.xlabel("Time of Day")
# plt.ylabel("Cumulative Profit ($)")
# plt.legend(loc='upper left')
# plt.grid(axis='y', linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, timedelta

# Create uniform datetime index
time_index = pd.date_range(start="2025-01-01 00:00", periods=288, freq="5min")
df["Time"] = time_index

# Set font sizes
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

### 1. EVs Arrived and Served ###
plt.figure(figsize=(10, 4))
plt.bar(df["Time"], df["EVs Arrived"], width=0.0015, color="skyblue", label="EVs Arrived")
plt.bar(df["Time"], df["EVs Served"], width=0.0015, color="orange", label="EVs Served")
plt.title("EVs Arrived and Served Over Time")
plt.xlabel("Time of Day")
plt.ylabel("EV Count")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.legend()
plt.tight_layout()
plt.savefig("figs/ev_arrival_served.png", dpi=300)
plt.close()

### 2. EV Queue and Electricity Spot ###
plt.figure(figsize=(10, 4))
plt.bar(df["Time"], df["Electricity Spot"], color="dodgerblue", label="Available Electricity Spot", width=0.0015)
plt.bar(df["Time"], [-q for q in df["EVs queue"]], color="firebrick", label="EV Queue Length", width=0.0015)
plt.axhline(0, color="black", linewidth=0.7)
plt.title("Charging Spot Availability and EV Queue Length")
plt.xlabel("Time of Day")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.legend()
plt.tight_layout()
plt.savefig("figs/ev_queue_spot.png", dpi=300)
plt.close()

### 3. HFCV Arrivals and Served ###
plt.figure(figsize=(10, 4))
plt.bar(df["Time"], df["HFCVs Arrived"], color="steelblue", width=0.0015, label="HFCVs Arrived")
plt.bar(df["Time"], df["HFCVs Served"], color="mediumseagreen", width=0.0015, label="HFCVs Served")
plt.title("HFCV Arrivals and Served Over Time")
plt.xlabel("Time of Day")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.legend()
plt.tight_layout()
plt.savefig("figs/hfcv_arrival_served.png", dpi=300)
plt.close()

### 4. Profit Decomposition ###
plt.figure(figsize=(10, 4))
plt.plot(df["Time"], df["Charging Station Profit"], color="black", label="Total Profit", linewidth=1.5)
plt.bar(df["Time"], df["Electricity Profit"], color="royalblue", width=0.0015, label="Electricity Profit")
plt.bar(df["Time"], df["Hydrogen Profit"], bottom=df["Electricity Profit"], color="limegreen", width=0.0015, label="Hydrogen Profit")
plt.title("Cumulative Profit Over Time")
plt.xlabel("Time of Day")
plt.ylabel("Cumulative Profit ($)")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.legend()
plt.tight_layout()
plt.savefig("figs/profit_decomposition.png", dpi=300)
plt.close()
