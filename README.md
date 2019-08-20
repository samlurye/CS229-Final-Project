# cs229-final-project: Exploring the evolution of modularity from scratch in neural networks

### This repository is built on a Python implementation of the NEAT algorithm found here: https://github.com/CodeReclaimers/neat-python.

In order to run a trial with MVG and fixed structure, execute
```
$ python evolve-feedforward.py config-fixedstruct-mvg
```

In order to run a trial with MVG and evolving structure, execute
```
$ python evolve-feedforward.py config-mvg
```

In order to run a trial with a fixed goal and fixed structure, execute
```
$ python evolve-feedforward.py config-fixedstruct-fixedgoal
```

In order to run a trial with a fixed goal and evolving structure, execute
```
$ python evolve-feedforward.py config-fixedgoal
```

In order to run a trial with a fixed goal, evolving structure, and connection cost, execute
```
$ python evolve-feedforward.py config-cc
```
