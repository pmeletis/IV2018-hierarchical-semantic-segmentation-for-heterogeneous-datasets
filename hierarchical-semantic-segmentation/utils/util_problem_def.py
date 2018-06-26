import json

# problem_def_path = '/home/panos/git/hierarchical-semantic-segmentation-2/semantic-segmentation/training_problem_def.json'

# problem_def_path = '/media/panos/data/datasets/mapillary/mapillary-vistas-dataset_public_v1.0/panos/jsons/problem01.json'

problem_def_path = '/media/panos/data/datasets/cityscapes/panos/jsons/problem03.json'

problem_def = json.load(open(problem_def_path))

cids2labels = problem_def['cids2labels']

for cid, label in enumerate(cids2labels):
  print(f"{cid:>2}", label)
