import lib.datasets.stanford as stanford
import lib.datasets.stanford_test as stanford_test
import lib.datasets.scannet as scannet
import lib.datasets.scannet_test as scannet_test
import lib.datasets.semantic3D as semantic3D
import lib.datasets.semantic3D_test as semantic3D_test

DATASETS = []


def add_datasets(module):
  DATASETS.extend([getattr(module, a) for a in dir(module) if 'Dataset' in a])


add_datasets(stanford)
add_datasets(stanford_test)
add_datasets(scannet)
add_datasets(scannet_test)
add_datasets(semantic3D)
add_datasets(semantic3D_test)


def load_dataset(name):
  '''Creates and returns an instance of the datasets given its name.
  '''
  # Find the model class from its name
  mdict = {dataset.__name__: dataset for dataset in DATASETS}
  if name not in mdict:
    print('Invalid dataset index. Options are:')
    # Display a list of valid dataset names
    for dataset in DATASETS:
      print('\t* {}'.format(dataset.__name__))
    raise ValueError(f'Dataset {name} not defined')
  DatasetClass = mdict[name]

  return DatasetClass
