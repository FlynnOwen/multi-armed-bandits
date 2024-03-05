<h1 align="center">
Multi Armed Bandits (MAB).
</h1>

The multi-armed bandit problem is a classic dilemma in probability theory and decision-making, often used to model situations where one must balance exploration (trying new options) and exploitation (leveraging known options). The name "multi-armed bandit" originates from the analogy of a gambler facing multiple slot machines (bandits), each with potentially different payoff probabilities. The objective is to maximize cumulative reward over time while facing uncertainty about which action (or bandit arm) yields the highest reward.

In its basic form, a player repeatedly chooses from a set of arms, each with an unknown reward distribution, and receives a stochastic reward associated with the chosen arm. The challenge lies in deciding which arm to pull at each step to maximize cumulative reward, taking into account the trade-off between exploring new arms to gather information and exploiting arms with potentially higher expected rewards based on current knowledge.

<h2  align="center">
Methods Overview
</h2>


<h2  align="center">
Usage
</h2>

Options exist for simulating both directly using the command line, or via passing a configuration file in the form of JSON. The second is recommended for reproducable simulation studies.

To see all possible simulation commands run
```bash
just help
```

or help for a specific command:
```bash
just {{COMMAND}} help
```

e.g:
```bash
just list-distributions help
```

To view possible distributions, and information about their associated parameters, run:

```bash
just list-distributions
```

<h4 align="center">
Simulating directly
</h4>

<p align="center">

To run a simulation by passing arguments directly, run:

```bash
just {{COMMAND}} {{*ARGS}}
```

Where `COMMAND` is a command returned from `just help`, and args are those required
by a specific command upon running `just {{COMMAND}} help`.
</p>

<h4 align="center">
Simulating from config (json) file
</h4>

<p align="center">

There is the option to perform a MAB simulation, reading arguments and configuration from a config file.
To perform this simulation, run:

```bash
just simulate-from-json {{COMMAND}} {{CONFIG}}
```
</p>

<h4 align="center">
Sample Outputs
</h4>

<p align="center">
<p align="center"> Metrics from the simulations are output to stdout. </p>
<img width="400" height="150" src="img/sim_metrics.png"/>
</p>

<p align="center">
<p align="center"> Plots are also optionally produced, detailing the the performance of the simulation. </p>
<img width="400" height="300" src="img/sim_cum_reward.png"/>
<img width="400" height="300" src="img/sim_pulls.png"/>
</p>

<p align="center">
<img width="400" height="300" src="img/sim_reward.png"/>
<img width="400" height="300" src="img/sim_residuals.png"/>
</p>