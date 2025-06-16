
# ODD Protocol Summary

## 1. Overview

**Purpose:**  
This agent-based simulation model is designed to evaluate the performance of an integrated hydrogen-electricity charging station that serves both EV and HFCV users. The objective is to assess how user behavior (based on delay and price sensitivity), infrastructure configurations, and renewable variability affect overall station performance, energy flows, and service outcomes.

**Entities, State Variables, and Scales:**  
- **Agents:**  
  - EV users (battery-based, price- and queue-sensitive)  
  - HFCV users (hydrogen-based, price- and availability-sensitive)  
- **Infrastructure components:**  
  - Charging spots (EV), refueling pumps (HFCV), electrolyzer, hydrogen tank, fuel cell, solar panel, wind turbine, battery
- **Temporal scale:**  
  - 5-minute discrete time steps over a 24-hour horizon  
- **Spatial scale:**  
  - Single station (non-spatially explicit)

**Process Overview and Scheduling:**  
At each time step, agents arrive probabilistically based on traffic demand profiles. They evaluate utility based on current wait time and energy price. If the utility is positive, they queue or begin service. The station updates energy production and storage states based on renewable generation and consumption, then logs outcomes.

## 2. Design Concepts

**Basic Principles:**  
The model is built on the principles of bounded rationality and utility-based decision-making under queueing delay and dynamic pricing. It extends multi-energy station modeling by integrating EV and HFCV behaviors with hybrid infrastructure.

**Emergence:**  
System-level metrics such as energy throughput, waiting times, and profitability emerge from individual agent decisions and real-time energy balancing.

**Adaptation:**  
Agents do not adapt across time steps but re-evaluate their decisions at each arrival based on current state conditions (myopic utility maximization).

**Objectives:**  
Agents attempt to maximize personal utility (based on price, wait time, and remaining energy need) when deciding whether to use the station.

**Sensing:**  
Agents observe current station conditions including wait times, spot availability, and pricing signals.

**Interaction:**  
Agents indirectly interact through shared infrastructure and queues.

**Stochasticity:**  
Arrival times, queue durations, and renewable generation follow stochastic profiles.

## 3. Details

**Initialization:**  
- EV and HFCV arrival profiles are generated using demand data from a 5-minute expanded dataset.  
- Initial battery, hydrogen, and queue states are set to empty or neutral.

**Input Data:**  
- `5-Minute_Expanded_Data.xlsx`: hourly demand, renewable generation profiles, and pricing schedules.

**Submodels:**  
- `simulation_updated.py`: Core simulation model  
- `simulation_experiments_1/2/3.py`: Sensitivity analyses on parameters such as spot availability, pricing, and storage sizing.

**Outputs:**  
- Service rates, total profit, energy usage, queue lengths, user acceptance ratios (saved in `/code/` as CSVs or logs)
