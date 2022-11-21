from neat.population import Population, CompleteExtinctionException
from neat.config import Config
from neat.six_util import iteritems, itervalues
from map import load_levels, get_area_mapa, get_new_map, get_unidades_dict
from random import random
from visualize import graph_per_stage
import time
class CarPopulation(Population):

    def run(self, fitness_function, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")


        total_levels = load_levels()
        # if generations equal  1 we choose the hardest lvl
        if n == 1:
            map_number = total_levels
        else:
            # otherwise we start from the easier
            map_number = 1
        map_path =  f'./models/levels/{map_number}.xml'
        unidades = get_unidades_dict(map_path)
        mapa = get_new_map(map_path)
        map_area = get_area_mapa(mapa)
        change_every = int(n / total_levels)
        k = 0
        

        start_time = time.time()
        generations_time = []
        while n is None or k < n:
            gen_time = time.time()

            if k != 0 and k % change_every == 0:
                map_number += 1
                map_path =  f'./models/levels/{map_number}.xml'
                unidades = get_unidades_dict(map_path)
                mapa = get_new_map(map_path)
                map_area = get_area_mapa(mapa)
                print(f'cambio de mapa ===> {map_number}')
                
                
            k += 1

            self.reporters.start_generation(self.generation)
            
            
            random_value = random()
            self.config.config_information = {"seed": random_value, "map_path": map_path, "map_area": map_area, "unidades_dict": unidades}
            # Evaluate all genomes using the user-provided function.
            fitness_function(list(iteritems(self.population)), self.config)

            # Gather and report statistics.
            best = None
            for g in itervalues(self.population):
                if best is None or g.fitness > best.fitness:
                    best = g

            generations_time.append(time.time() - gen_time)
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            if k % change_every == 0:
                graph_per_stage(self.config, map_number, best, self.reporters.reporters[0], n, total_levels, generations_time)
                generations_time = []

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in itervalues(self.population))
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)
        print(f"Total time --- {time.time() - start_time} seconds ---")
        return self.best_genome

class CarConfig(Config):
    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename, config_information=None):
        super().__init__(genome_type, reproduction_type, species_set_type, stagnation_type, filename)
        self.config_information = config_information