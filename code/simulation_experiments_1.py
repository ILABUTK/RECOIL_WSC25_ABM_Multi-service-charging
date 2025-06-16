import pandas as pd
import numpy as np
from simulation_updated import ChargingStation, step_durtion

experiments = [
    {"name": "low_price_case_5", "price_rate": 0.5},
    {"name": "low_price_case_4", "price_rate": 0.6},
    {"name": "low_price_case_3", "price_rate": 0.7},
    {"name": "low_price_case_2", "price_rate": 0.8},
    {"name": "low_price_case_1", "price_rate": 0.9},
    {"name": "base_case", "price_rate": 1.0},
    {"name": "high_price_case_1", "price_rate": 1.1},
    {"name": "high_price_case_2", "price_rate": 1.2},
    {"name": "high_price_case_3", "price_rate": 1.3},
    {"name": "high_price_case_4", "price_rate": 1.4},
    {"name": "high_price_case_5", "price_rate": 1.5},
]

# Store all results
all_results = []

for exp in experiments:
    profit_list = []
    served_list = []
    arrived_list = []
    fully_charged_list = []
    wait_time_list = []
    charge_time_list = []
    charge_amount_list = []

    for seed in range(10):
        np.random.seed(seed)
        model = ChargingStation(
            E_spot=4,
            H_spot=2,
            lambda_ev=8,
            lambda_hfcv=8,
            hfcv_refuel_time=step_durtion,
            price_rate=exp["price_rate"]
        )
        for i in range(288):
            model.step(i)

        df_result = pd.DataFrame(model.data)

        total_profit = df_result['Charging Station Profit'][287]
        total_served = sum(df_result['EVs Served'])
        total_arrived = sum(df_result['EVs Arrived'])
        total_fully_charged = sum(df_result['EVs Fully Charged'])
        avg_wait_time = (
            sum(df_result["Arrived EVs Total Wait Time"]) / total_arrived if total_arrived > 0 else 0
        )
        avg_charge_time = (
            sum(df_result["Served EVs Total Charging Time"]) / total_served if total_served > 0 else 0
        )
        total_charge_amount = (
            sum(df_result["Total EVs Charged Amount"])
        )

        profit_list.append(total_profit)
        served_list.append(total_served)
        arrived_list.append(total_arrived)
        fully_charged_list.append(total_fully_charged)
        wait_time_list.append(avg_wait_time)
        charge_time_list.append(avg_charge_time)
        charge_amount_list.append(total_charge_amount)

    # Append the average results for this experiment
    all_results.append({
        "Experiment": exp["name"],
        "Price Rate": exp["price_rate"],
        "Avg Profit": round(np.mean(profit_list), 2),
        "Avg EVs Served": round(np.mean(served_list), 2),
        "Avg EVs Arrived": round(np.mean(arrived_list), 2),
        "Avg Fully Charged EVs": round(np.mean(fully_charged_list), 2),
        "Avg Wait Time (min)": round(np.mean(wait_time_list), 2),
        "Avg Charge Time (min)": round(np.mean(charge_time_list), 2),
        "Avg Total Charge Amount (kW)": round(np.mean(charge_amount_list), 2),
    })

# Convert to DataFrame and display
summary_df = pd.DataFrame(all_results)
print(summary_df.to_string(index=False))

# Save the summary DataFrame to Excel
summary_df.to_excel("experiment_summary_1.xlsx", index=False)
print("Results saved to experiment_summary.xlsx")
