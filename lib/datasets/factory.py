# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.pascal_voc_cyclewater import pascal_voc_cyclewater
from datasets.pascal_voc_cycleclipart import pascal_voc_cycleclipart
from datasets.sim10k import sim10k
from datasets.water import water
from datasets.clipart import clipart
from datasets.comic import comic
from datasets.amds import amds
from datasets.sim10k_cycle import sim10k_cycle
from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.foggy_cityscape import foggy_cityscape

import numpy as np
import os

for split in ['train', 'trainval','val','test']:
  name = 'cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape(split))
for split in ['train', 'trainval','val','test']:
  name = 'cityscape_car_{}'.format(split)
  __sets[name] = (lambda split=split : cityscape_car(split))
for split in ['train', 'trainval','test']:
  name = 'foggy_cityscape_{}'.format(split)
  __sets[name] = (lambda split=split : foggy_cityscape(split))
for split in ['train','val']:
  name = 'sim10k_{}'.format(split)
  __sets[name] = (lambda split=split : sim10k(split))
for split in ['train', 'val']:
  name = 'sim10k_cycle_{}'.format(split)
  __sets[name] = (lambda split=split: sim10k_cycle(split))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_water_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc_water(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_cycleclipart_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cycleclipart(split, year))
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
      name = 'voc_cyclewater_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: pascal_voc_cyclewater(split, year))
for split in ['train', 'trainval', 'test']:
  for data_percentage in ['', '_1_00', '_1_01', '_1_02']:
    dataset_name = 'clipart{}'.format(data_percentage)
    name = '{}_{}'.format(dataset_name, split)
    __sets[name] = (lambda split=split, dataset_name=dataset_name: clipart(split,devkit_path=os.path.join('datasets', dataset_name)))
for split in ['train', 'test']:
  for data_percentage in ['', '_1_00', '_1_01', '_1_02']:
    dataset_name = 'comic{}'.format(data_percentage)
    name = '{}_{}'.format(dataset_name, split)
    __sets[name] = (lambda split=split, dataset_name=dataset_name: comic(split,devkit_path=os.path.join('datasets', dataset_name)))
for split in ['train', 'test']:
  for data_percentage in ['', '_1_00', '_1_01', '_1_02']:
    dataset_name = 'watercolor{}'.format(data_percentage)
    name = '{}_{}'.format(dataset_name, split)
    __sets[name] = (lambda split=split, dataset_name=dataset_name: water(split,devkit_path=os.path.join('datasets', dataset_name)))
for split in ['train', 'test']:
  dataset_name = 'amds'
  name = '{}_{}'.format(dataset_name, split)
  __sets[name] = (lambda split=split, dataset_name=dataset_name: amds(split,devkit_path=os.path.join('datasets', dataset_name)))
def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
