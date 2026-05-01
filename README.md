# HIVE

This repository provides the official implementation of the paper:

> **HIVE: A Hypergraph-based game-theoretic Interactive Value decomposition Engine for multi-lateral agents collaboration**




# Extended Python MARL framework - EPyMARL

## How to run HIVE?
After installing all of the dependencies and setting the working directory to src, you can run the following command lines:
```shell
python main.py --config=HIVE --env-config=sc2 with env_args.map_name=MMM2 sample_size=10 t_max=2050000 hyper_edge_num=16
```

## Matrix Game Experiments:
```shell
python main.py --config=HIVE --env-config=gymma with env_args.time_limit=1 env_args.key="matrixgames:climbing-nostate-v0" hyper_edge_num=16 epsilon_anneal_time=500000
```

## Dependencies

- **SMAC (StarCraft Multi-Agent Challenge)**:  
  https://github.com/andrewmarx/smac  

- **SMACv2**:  
  https://github.com/oxwhirl/smacv2  

Please follow the official instructions in the above repositories to install the environments.


## 🎬 Visualization Results

We provide qualitative demonstrations...

### SMAC: 1c3s5z
HIVE learns coordinated target selection and positioning.

<video src="results/SMAC-1c3s5z.mp4" controls width="600"></video>

### SMACv2: terran_10_vs_10
HIVE exhibits stable large-scale coordination.

<video src="results/SMACv2-terran10.mp4" controls width="600"></video>




