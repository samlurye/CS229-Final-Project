import os
import neat
import visualize
import numpy as np
import random
import pickle

class Task:

    def __init__(self, config_file, goal_fns, goal_freq, generations):
        self.config = neat.Config(neat.genome.LayeredGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
        self.pop = neat.Population(self.config)
        self.goal_fns = goal_fns
        inputs = Task.get_binary_inputs(self.config.genome_config.num_inputs)
        outputs = [list(map(goal_fn, inputs)) for goal_fn in self.goal_fns]
        self.inputs = np.array(inputs).transpose(1, 0)
        self.outputs = [np.array(o).transpose(1, 0) for o in outputs]
        self.generations = generations
        self.ctr = 0
        self.goal_freq = goal_freq
        self.current_goal = 0

    @staticmethod
    def get_binary_inputs(n_bits):
        tmp = [list(format(i, f"0{n_bits}b")) for i in range(2 ** n_bits)]
        return [list(map(float, lst)) for lst in tmp] 

    def eval_genomes(self, genomes, config):
        outputs = self.outputs[self.current_goal]
        use_cc = random.random() < 0.
        for genome_id, genome in genomes:
            genome.fitness = self.inputs.shape[1]
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            output = net.activate(self.inputs)
            output = [np.round(o) for o in output]
            n_incorrect = np.sum(np.abs(output - outputs))
            genome.fitness -= n_incorrect
            # if n_incorrect < 12:
            #     genome.fitness = 256
            # else:
            #     genome.fitness -= n_incorrect
            #     if use_cc:
            #         genome.fitness -= sum([1 for cg in genome.connections.values() if cg.enabled])
        self.ctr += 1
        if not self.ctr % self.goal_freq:
            self.current_goal += 1
            self.current_goal %= len(self.goal_fns)

    def run(self):
        self.pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.pop.add_reporter(stats)
        mod = neat.reporting.ModularityReporter(10)
        self.pop.add_reporter(mod)

        winner = self.pop.run(lambda x, y: self.eval_genomes(x, y), self.generations)

        winner_mod = winner.modularity()
        mod.means[-1] = winner_mod

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))
        print("Modularity: {:.4f}".format(winner_mod))

        file_stem = "Experiments/layered-fixedgoal/trial2"

        node_names = {-(i+1): f"x{i}" for i in range(self.config.genome_config.num_inputs)}
        node_names.update({i: f"y{i}" for i in range(self.config.genome_config.num_outputs)})
        visualize.draw_net(self.config, winner, node_names=node_names, filename=file_stem+".gv")

        save_stats = {
        	"f_best": np.array([g.fitness for g in stats.most_fit_genomes]),
        	"f_mean": np.array(stats.get_fitness_mean()),
        	"f_stddev": np.array(stats.get_fitness_stdev()),
        	"m_best": np.array(mod.best),
        	"m_mean": np.array(mod.means),
        	"m_stddev": np.array(mod.stddevs)
        }

        with open(file_stem + ".pkl", "wb") as f:
        	pickle.dump(save_stats, f, pickle.HIGHEST_PROTOCOL)


def xor_or(args):
    return (float((args[0] != args[1]) or (args[2] != args[3])),)

def xor_and(args):
    return (float((args[0] != args[1]) and (args[2] != args[3])),)

def retina_left(args):
    return (args[0] and args[1] and args[2] and args[3]) \
        or (args[0] and args[1] and args[2] and not args[3]) \
        or (args[0] and not args[1] and args[2] and not args[3]) \
        or (args[0] and not args[1] and args[2] and args[3]) \
        or (args[0] and not args[1] and not args[2] and not args[3]) \
        or (args[0] and args[1] and not args[2] and args[3]) \
        or (not args[0] and not args[1] and args[2] and not args[3]) \
        or (not args[0] and args[1] and args[2] and args[3])

def retina_right(args):
    return (args[0] and args[1] and args[2] and args[3]) \
        or (args[0] and args[1] and args[2] and not args[3]) \
        or (args[0] and not args[1] and args[2] and args[3]) \
        or (not args[0] and args[1] and not args[2] and args[3]) \
        or (args[0] and args[1] and not args[2] and args[3]) \
        or (not args[0] and args[1] and not args[2] and not args[3]) \
        or (not args[0] and args[1] and args[2] and args[3]) \
        or (not args[0] and not args[1] and not args[2] and args[3])

def retina_and(args):
    return (float(retina_left(args[:4]) and retina_right(args[4:])),)

def retina_or(args):
    return (float(retina_left(args[:4]) or retina_right(args[4:])),)

if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')

    task = Task(config_path, [retina_and], 10000, 10000)
    task.run()


