import networkx
import random

cifar100_theory = """aquatic_mammals1 <= beaver, dolphin
aquatic_mammals2 <= otter, seal
aquatic_mammals12 <= aquatic_mammals1, aquatic_mammals2
aquatic_mammals3 <= aquatic_mammals12, whale
fish1 <= aquarium_fish, flatfish
fish2 <= ray, shark
fish12 <= fish1, fish2
fish3 <= fish12, trout
flowers1 <= orchid, poppy
flowers2 <= rose, sunflower
flowers12 <= flowers1, flowers2
flowers3 <= flowers12, tulip
food_containers1 <= bottle, bowl
food_containers2 <= can, cup
food_containers12 <= food_containers1, food_containers2
food_containers3 <= food_containers12, plate
fruit_and_vegetables1 <= apple, mushroom
fruit_and_vegetables2 <= orange, pear
fruit_and_vegetables12 <= fruit_and_vegetables1, fruit_and_vegetables2
fruit_and_vegetables3 <= fruit_and_vegetables12, sweet_pepper
household_electrical_devices1 <= clock, keyboard
household_electrical_devices2 <= lamp, telephone
household_electrical_devices12 <= household_electrical_devices1, household_electrical_devices2
household_electrical_devices3 <= household_electrical_devices12, television
household_furniture1 <= bed, chair
household_furniture2 <= couch, table
household_furniture12 <= household_furniture1, household_furniture2
household_furniture3 <= household_furniture12, wardrobe
insects1 <= bee, beetle
insects2 <= butterfly, caterpillar
insects12 <= insects1, insects2
insects3 <= insects12, cockroach
large_carnivores1 <= bear, leopard
large_carnivores2 <= lion, tiger
large_carnivores12 <= large_carnivores1, large_carnivores2
large_carnivores3 <= large_carnivores12, wolf
large_man_made_outdoor_things1 <= bridge, castle
large_man_made_outdoor_things2 <= house, road
large_man_made_outdoor_things12 <= large_man_made_outdoor_things1, large_man_made_outdoor_things2
large_man_made_outdoor_things3 <= large_man_made_outdoor_things12, skyscraper
large_natural_outdoor_scenes1 <= cloud, forest
large_natural_outdoor_scenes2 <= mountain, plain
large_natural_outdoor_scenes12 <= large_natural_outdoor_scenes1, large_natural_outdoor_scenes2
large_natural_outdoor_scenes3 <= large_natural_outdoor_scenes12, sea
large_omnivores_and_herbivores1 <= camel, cattle
large_omnivores_and_herbivores2 <= chimpanzee, elephant
large_omnivores_and_herbivores12 <= large_omnivores_and_herbivores1, large_omnivores_and_herbivores2
large_omnivores_and_herbivores3 <= large_omnivores_and_herbivores12, kangaroo
medium_sized_mammals1 <= fox, porcupine
medium_sized_mammals2 <= possum, raccoon
medium_sized_mammals12 <= medium_sized_mammals1, medium_sized_mammals2
medium_sized_mammals3 <= medium_sized_mammals12, skunk
non_insect_invertebrates1 <= crab, lobster
non_insect_invertebrates2 <= snail, spider
non_insect_invertebrates12 <= non_insect_invertebrates1, non_insect_invertebrates2
non_insect_invertebrates3 <= non_insect_invertebrates12, worm
people1 <= baby, boy
people2 <= girl, man
people12 <= people1, people2
people3 <= people12, woman
reptiles1 <= crocodile, dinosaur
reptiles2 <= lizard, snake
reptiles12 <= reptiles1, reptiles2
reptiles3 <= reptiles12, turtle
small_mammals1 <= hamster, mouse
small_mammals2 <= rabbit, shrew
small_mammals12 <= small_mammals1, small_mammals2
small_mammals3 <= small_mammals12, squirrel
trees1 <= maple_tree, oak_tree
trees2 <= palm_tree, pine_tree
trees12 <= trees1, trees2
trees3 <= trees12, willow_tree
vehiclesA1 <= bicycle, bus
vehiclesA2 <= motorcycle, pickup_truck
vehiclesA12 <= vehiclesA1, vehiclesA2
vehiclesA3 <= vehiclesA12, train
vehiclesB1 <= lawn_mower, rocket
vehiclesB2 <= streetcar, tank
vehiclesB12 <= vehiclesB1, vehiclesB2
vehiclesB3 <= vehiclesB12, tractor
parent1 <= aquatic_mammals3, fish3
parent2 <= flowers3, food_containers3
parent3 <= fruit_and_vegetables3, household_electrical_devices3 
parent4 <= household_furniture3, insects3
parent5 <= large_carnivores3, large_man_made_outdoor_things3
parent6 <= large_natural_outdoor_scenes3, large_omnivores_and_herbivores3
parent7 <= medium_sized_mammals3, non_insect_invertebrates3
parent8 <= people3, reptiles3
parent9 <= small_mammals3, trees3 
parent10 <= vehiclesA3, vehiclesB3 
superparent1 <= parent1, parent2
superparent2 <= parent3, parent4
superparent3 <= parent5, parent6
superparent4 <= parent7, parent8
superparent5 <= parent9, parent10
greatparent1 <= superparent1, superparent2
greatparent2 <= superparent3, superparent4
greatparent12 <= greatparent1, greatparent2
entities <= greatparent12, superparent5
"""

cifar10_theory = """land_transportation <= automobile, truck
other_transportation <= airplane, ship
transportation <= land_transportation, other_transportation
home_land_animals <= cat, dog 
wild_land_animals <= deer, horse 
land_animals <= home_land_animals, wild_land_animals
other_animals <= bird, frog 
animals <= land_animals, other_animals 
entities <= transportation, animals  
"""

cifar10_output_classes = [
    "land_transportation",
    "other_transportation",
    "transportation",
    "home_land_animals",
    "wild_land_animals",
    "land_animals",
    "other_animals",
    "animals",
    "entities",
    "automobile",
    "truck",
    "airplane",
    "ship",
    "cat",
    "dog",
    "horse",
    "deer",
    "bird",
    "frog",
]

cifar10_classes = [
    "automobile",
    "truck",
    "airplane",
    "ship",
    "cat",
    "deer",
    "dog",
    "horse",
    "bird",
    "frog",
]

cifar100_output_classes = [
    "aquatic_mammals1",
    "aquatic_mammals2",
    "aquatic_mammals12",
    "aquatic_mammals3",
    "fish1",
    "fish2",
    "fish12",
    "fish3",
    "flowers1",
    "flowers2",
    "flowers12",
    "flowers3",
    "food_containers1",
    "food_containers2",
    "food_containers12",
    "food_containers3",
    "fruit_and_vegetables1",
    "fruit_and_vegetables2",
    "fruit_and_vegetables12",
    "fruit_and_vegetables3",
    "household_electrical_devices1",
    "household_electrical_devices2",
    "household_electrical_devices12",
    "household_electrical_devices3",
    "household_furniture1",
    "household_furniture2",
    "household_furniture12",
    "household_furniture3",
    "insects1",
    "insects2",
    "insects12",
    "insects3",
    "large_carnivores1",
    "large_carnivores2",
    "large_carnivores12",
    "large_carnivores3",
    "large_man_made_outdoor_things1",
    "large_man_made_outdoor_things2",
    "large_man_made_outdoor_things12",
    "large_man_made_outdoor_things3",
    "large_natural_outdoor_scenes1",
    "large_natural_outdoor_scenes2",
    "large_natural_outdoor_scenes12",
    "large_natural_outdoor_scenes3",
    "large_omnivores_and_herbivores1",
    "large_omnivores_and_herbivores2",
    "large_omnivores_and_herbivores12",
    "large_omnivores_and_herbivores3",
    "medium_sized_mammals1",
    "medium_sized_mammals2",
    "medium_sized_mammals12",
    "medium_sized_mammals3",
    "non_insect_invertebrates1",
    "non_insect_invertebrates2",
    "non_insect_invertebrates12",
    "non_insect_invertebrates3",
    "people1",
    "people2",
    "people12",
    "people3",
    "reptiles1",
    "reptiles2",
    "reptiles12",
    "reptiles3",
    "small_mammals1",
    "small_mammals2",
    "small_mammals12",
    "small_mammals3",
    "trees1",
    "trees2",
    "trees12",
    "trees3",
    "vehiclesA1",
    "vehiclesA2",
    "vehiclesA12",
    "vehiclesA3",
    "vehiclesB1",
    "vehiclesB2",
    "vehiclesB12",
    "vehiclesB3",
    "parent1",
    "parent2",
    "parent3",
    "parent4",
    "parent5",
    "parent6",
    "parent7",
    "parent8",
    "parent9",
    "parent10",
    "superparent1",
    "superparent2",
    "superparent3",
    "superparent4",
    "superparent5",
    "greatparent1",
    "greatparent2",
    "greatparent12",
    "entities",
    "beaver",
    "dolphin",
    "otter",
    "seal",
    "whale",
    "bear",
    "leopard",
    "lion",
    "tiger",
    "wolf",
    "camel",
    "cattle",
    "chimpanzee",
    "elephant",
    "kangaroo",
    "fox",
    "porcupine",
    "possum",
    "raccoon",
    "skunk",
    "hamster",
    "mouse",
    "rabbit",
    "shrew",
    "squirrel",
    "baby",
    "boy",
    "girl",
    "man",
    "woman",
    "crab",
    "lobster",
    "crocodile",
    "dinosaur",
    "lizard",
    "snake",
    "turtle",
    "aquarium_fish",
    "flatfish",
    "ray",
    "shark",
    "trout",
    "bee",
    "beetle",
    "butterfly",
    "caterpillar",
    "cockroach",
    "snail",
    "spider",
    "worm",
    "orchid",
    "poppy",
    "rose",
    "sunflower",
    "tulip",
    "maple_tree",
    "oak_tree",
    "palm_tree",
    "pine_tree",
    "willow_tree",
    "apple",
    "mushroom",
    "orange",
    "pear",
    "sweet_pepper",
    "clock",
    "keyboard",
    "lamp",
    "telephone",
    "television",
    "bed",
    "chair",
    "couch",
    "table",
    "wardrobe",
    "bottle",
    "bowl",
    "can",
    "cup",
    "plate",
    "bicycle",
    "bus",
    "motorcycle",
    "pickup_truck",
    "train",
    "lawn_mower",
    "rocket",
    "streetcar",
    "tank",
    "tractor",
    "bridge",
    "castle",
    "house",
    "road",
    "skyscraper",
    "cloud",
    "forest",
    "mountain",
    "plain",
    "sea",
]

cifar100_classes = [
    "beaver",
    "dolphin",
    "otter",
    "seal",
    "whale",
    "bear",
    "leopard",
    "lion",
    "tiger",
    "wolf",
    "camel",
    "cattle",
    "chimpanzee",
    "elephant",
    "kangaroo",
    "fox",
    "porcupine",
    "possum",
    "raccoon",
    "skunk",
    "hamster",
    "mouse",
    "rabbit",
    "shrew",
    "squirrel",
    "baby",
    "boy",
    "girl",
    "man",
    "woman",
    "crab",
    "lobster",
    "crocodile",
    "dinosaur",
    "lizard",
    "snake",
    "turtle",
    "aquarium_fish",
    "flatfish",
    "ray",
    "shark",
    "trout",
    "bee",
    "beetle",
    "butterfly",
    "caterpillar",
    "cockroach",
    "snail",
    "spider",
    "worm",
    "orchid",
    "poppy",
    "rose",
    "sunflower",
    "tulip",
    "maple_tree",
    "oak_tree",
    "palm_tree",
    "pine_tree",
    "willow_tree",
    "apple",
    "mushroom",
    "orange",
    "pear",
    "sweet_pepper",
    "clock",
    "keyboard",
    "lamp",
    "telephone",
    "television",
    "bed",
    "chair",
    "couch",
    "table",
    "wardrobe",
    "bottle",
    "bowl",
    "can",
    "cup",
    "plate",
    "bicycle",
    "bus",
    "motorcycle",
    "pickup_truck",
    "train",
    "lawn_mower",
    "rocket",
    "streetcar",
    "tank",
    "tractor",
    "bridge",
    "castle",
    "house",
    "road",
    "skyscraper",
    "cloud",
    "forest",
    "mountain",
    "plain",
    "sea",
]
cached_base_classes = dict()


def find_base_classes(superclass, is_a_graph, all_base_classes):
    if superclass == "None":
        return all_base_classes
    if superclass not in cached_base_classes:
        if not is_a_graph.has_node(superclass):
            cached_base_classes[superclass] = [superclass]
        cached_base_classes[superclass] = [
            seed
            for seed in all_base_classes
            if is_a_graph.has_node(seed)
            and networkx.has_path(is_a_graph, seed, superclass)
        ]
    return cached_base_classes[superclass]


def create_is_a_graph(str_hierarchies):
    is_a_graph = networkx.DiGraph()
    for line in str_hierarchies.splitlines():
        split = line.split("<=")
        parent = split[0]
        children = split[1].split(", ")
        for child in children:
            is_a_graph.add_edge(child.strip(), parent.strip())
    return is_a_graph


def get_all_ancestors(object_type, is_a_graph):
    superclass = object_type
    superclasses = list()
    while True:
        # Pick a dest edge from current node
        outgoing_edges = list(is_a_graph.out_edges(superclass))
        if outgoing_edges:
            edge_id = random.randint(0, len(outgoing_edges) - 1)
            dest = outgoing_edges[edge_id][1]
            superclasses.append(dest)
            superclass = dest
        else:
            return superclasses
