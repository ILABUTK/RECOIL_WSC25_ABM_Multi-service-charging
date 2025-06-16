import pandas as pd
from simulation_updated import ChargingStation, step_durtion

# Define experiment configurations for varying arrival rates and spot capacities
spot_sufficiency_experiments = [
    {"name": f"arr_{arr}_spot_{spot}", "lambda_ev": arr, "lambda_hfcv": arr, "ev_spots": spot, "hfcv_spots": int(spot / 2)}
    for arr in [4, 6, 8, 10, 12]
    for spot in [2, 4, 6, 8]
]

# Store results
results = []

# Run experiments 10 times for each setting
for exp in spot_sufficiency_experiments:
    avg_metrics = {
        "Experiment": exp["name"],
        "EV Arrival Rate": exp["lambda_ev"],
        "HFCV Arrival Rate": exp["lambda_hfcv"],
        "EV Spots": exp["ev_spots"],
        "HFCV Spots": exp["hfcv_spots"],
        "Avg Profit": 0,
        "EVs Arrived": 0,
        "EVs Served": 0,
        "EVs Fully Charged": 0,
        "Avg EV Waiting Time": 0,
        "Avg EV Charging Time": 0,
        "EVs Charged Amount": 0,
        "HFCVs Arrived": 0,
        "HFCVs Served": 0,
        "HFCVs Fully Charged": 0,
    }

    for _ in range(10):
        model = ChargingStation(
            E_spot=exp["ev_spots"],
            H_spot=exp["hfcv_spots"],
            lambda_ev=exp["lambda_ev"],
            lambda_hfcv=exp["lambda_hfcv"],
            hfcv_refuel_time=step_durtion
        )

        for i in range(288):  # 24 hours * 12 (every 5 minutes)
            model.step(i)

        df_result = pd.DataFrame(model.data)

        avg_metrics["Avg Profit"] += df_result['Charging Station Profit'][287] / 10
        avg_metrics["EVs Arrived"] += sum(df_result['EVs Arrived']) / 10
        avg_metrics["EVs Served"] += sum(df_result['EVs Served']) / 10
        avg_metrics["EVs Fully Charged"] += sum(df_result['EVs Fully Charged']) / 10
        avg_metrics["Avg EV Waiting Time"] += (sum(df_result["Arrived EVs Total Wait Time"]) / sum(df_result['EVs Arrived'])) / 10 if sum(df_result['EVs Arrived']) else 0
        avg_metrics["Avg EV Charging Time"] += (sum(df_result["Served EVs Total Charging Time"]) / sum(df_result['EVs Served'])) / 10 if sum(df_result['EVs Served']) else 0
        avg_metrics["EVs Charged Amount"] += sum(df_result['Total EVs Charged Amount']) / 10
        avg_metrics["HFCVs Arrived"] += sum(df_result['HFCVs Arrived']) / 10
        avg_metrics["HFCVs Served"] += sum(df_result['HFCVs Served']) / 10
        avg_metrics["HFCVs Fully Charged"] += sum(df_result['HFCVs Fully Charged']) / 10

    results.append(avg_metrics)

# Save all results
df_final = pd.DataFrame(results)
print(df_final.to_string(index=False))
df_final.to_excel("experiment_summary_2.xlsx", index=False)
