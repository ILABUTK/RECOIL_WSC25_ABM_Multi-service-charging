import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa import Model

class EV:
    def __init__(self, model, capacity, initial_charge, charge_speed):
        self.charging = False
        self.capacity = capacity
        self.remaining_charge = np.random.uniform(0, capacity) if initial_charge is None else initial_charge
        self.charge_speed = charge_speed
        self.charge_time = model.ev_charge_time
        self.time_left = self.charge_time
    
    def start_charging(self, price, charge_level, available_stations):
        self.charging = available_stations > 0
        self.charge_amount = self.utility_function(price, charge_level, available_stations)
    
    def utility_function(self, price, charge_level, available_stations):
        if available_stations == 0:
            return 0  # No available station, no charging
        if charge_level > 0.5:
            return self.capacity - self.remaining_charge  # Charge fully
        elif price <= 3:
            return self.capacity - self.remaining_charge  # Charge fully
        else:
            return 0  # Do not charge

class HFCV:
    def __init__(self, model, capacity, initial_charge, charge_speed):
        self.refueling = False
        self.capacity = capacity
        self.remaining_charge = np.random.uniform(0, capacity) if initial_charge is None else initial_charge
        self.charge_speed = charge_speed
        self.refuel_time = model.hfcv_refuel_time
        self.time_left = self.refuel_time
    
    def start_refueling(self, price, charge_level, available_stations):
        self.refueling = available_stations > 0 or charge_level > 0.8
        self.refuel_amount = self.utility_function(price, charge_level, available_stations)
    
    def utility_function(self, price, charge_level, available_stations):
        if available_stations == 0 and charge_level <= 0.8:
            return 0  # No available station and charge level not high enough, do not refuel
        if charge_level > 0.5:
            return self.capacity - self.remaining_charge  # Refuel fully
        elif price <= 3:
            return self.capacity - self.remaining_charge  # Refuel fully
        else:
            return 0  # Do not refuel


class ChargingStation(Model):
    def __init__(self, lambda_ev, lambda_hfcv, ev_charge_time, hfcv_refuel_time, steps, 
                 solar_efficiency=0.2, wind_efficiency=0.3, elec_to_hydrogen_rate=0.7, hydrogen_to_elec_rate=0.6,
                 elec_storage_capacity=500, hydrogen_storage_capacity=300):
        super().__init__()
        self.num_hydrogen_spots = 5
        self.num_electric_spots = 10
        self.ev_queue = []
        self.hfcv_queue = []
        self.steps = steps
        self.ev_charge_time = ev_charge_time
        self.hfcv_refuel_time = hfcv_refuel_time
        
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
        self.lambda_ev = lambda_ev
        self.lambda_hfcv = lambda_hfcv
        
        self.data = {
            "EVs Arrived": [],
            "HFCVs Arrived": [],
            "EVs Served": [],
            "HFCVs Served": [],
            "Charging Station Profit": [],
            "Electricity Storage": [],
            "Hydrogen Storage": []
        }
    
    def generate_solar_energy(self, solar_level):
        generated = solar_level * self.solar_efficiency
        self.store_electricity(generated)
        return generated
    
    def generate_wind_energy(self, wind_level):
        generated = wind_level * self.wind_efficiency
        self.store_electricity(generated)
        return generated
    
    def electricity_to_hydrogen(self, electricity):
        hydrogen_produced = electricity * self.elec_to_hydrogen_rate
        self.store_hydrogen(hydrogen_produced)
        self.electricity_storage -= electricity
        return hydrogen_produced
    
    def hydrogen_to_electricity(self, hydrogen):
        electricity_produced = hydrogen * self.hydrogen_to_elec_rate
        self.store_electricity(electricity_produced)
        self.hydrogen_storage -= hydrogen
        return electricity_produced
    
    def store_electricity(self, amount):
        self.electricity_storage = min(self.electricity_storage + amount, self.elec_storage_capacity)
    
    def store_hydrogen(self, amount):
        self.hydrogen_storage = min(self.hydrogen_storage + amount, self.hydrogen_storage_capacity)
    
    def step(self):
        # Simulating energy generation
        solar_input = np.random.uniform(0, 10)
        wind_input = np.random.uniform(0, 10)
        self.generate_solar_energy(solar_input)
        self.generate_wind_energy(wind_input)
        
        # Simulating vehicle arrivals
        num_new_evs = np.random.poisson(self.lambda_ev)
        num_new_hfcvs = np.random.poisson(self.lambda_hfcv)
        
        self.data["EVs Arrived"].append(num_new_evs)
        self.data["HFCVs Arrived"].append(num_new_hfcvs)
        
        charging_count = min(num_new_evs, self.num_electric_spots)
        refueling_count = min(num_new_hfcvs, self.num_hydrogen_spots)
        
        self.data["EVs Served"].append(charging_count)
        self.data["HFCVs Served"].append(refueling_count)
        
        profit = (charging_count * 5) + (refueling_count * 3)
        self.data["Charging Station Profit"].append(profit)
        
        # Store energy states
        self.data["Electricity Storage"].append(self.electricity_storage)
        self.data["Hydrogen Storage"].append(self.hydrogen_storage)


# Run the simulation
model = ChargingStation(lambda_ev=10, lambda_hfcv=7, ev_charge_time=30, hfcv_refuel_time=5, steps=100)

for i in range(100):
    model.step()

df = pd.DataFrame(model.data)

# Plot EV Arrivals and Served
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['EVs Arrived'], label='EVs Arrived')
plt.plot(df.index, df['EVs Served'], label='EVs Served')
plt.title("Number of Arriving and Served EVs")
plt.xlabel("Time Steps")
plt.ylabel("Count")
plt.legend()
plt.show()

# Plot HFCV Arrivals and Served
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['HFCVs Arrived'], label='HFCVs Arrived')
plt.plot(df.index, df['HFCVs Served'], label='HFCVs Served')
plt.title("Number of Arriving and Served HFCVs")
plt.xlabel("Time Steps")
plt.ylabel("Count")
plt.legend()
plt.show()

# Plot Charging Station Profit
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Charging Station Profit'], label='Charging Station Profit', color='green')
plt.title("Charging Station Profit Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Profit ($)")
plt.legend()
plt.show()
