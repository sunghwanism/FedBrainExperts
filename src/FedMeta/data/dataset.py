import abc
import functools
import os

import torch

from .utils import rates_to_latency_code, rates_to_population_code



class ExtendedDataset(abc.ABC, torch.utils.data.Dataset):
    """
    Interface extending torch Dataset with additional attributes.
    """
    _extended_keys = ["input_dim", "output_dim", "type"]

    def __init__(self, **kwargs):
        super().__init__()
        for key in self._extended_keys:
            setattr(self, key, kwargs[key])


def extended_dataset(dataset, **kwargs):
    """
    Monkey patch existing torch Dataset objects to adhere to the ExtendedDataset interface.
    """
    for key in ExtendedDataset._extended_keys:
        setattr(dataset, key, kwargs[key])

    return dataset


class Classification(ExtendedDataset):
    """
    Classification dataset given a list of functions [f_1(x), f_2(x), ... f_n(x)] that each define one class.
    """
    def __init__(self, input_dim, funs_list, num_samples_per_class, spiking=False):
        super().__init__(input_dim=input_dim, output_dim=len(funs_list), type="classification")

        self.funs_list = funs_list
        self.num_classes = len(funs_list)
        self.num_samples = num_samples_per_class * self.num_classes

        # Sample points from each function
        self.data, self.target = [], []
        for f, fun in enumerate(self.funs_list):
            x, y = fun.sample(num_samples_per_class)
            self.data.append(y)
            self.target.append(f * torch.ones(num_samples_per_class, dtype=int))

        # Shuffle task data and concatenate into tensor
        permutation = torch.randperm(self.num_samples)
        self.data = torch.cat(self.data)[permutation]
        self.target = torch.cat(self.target)[permutation]

        if spiking:
            self.data = rates_to_latency_code(self.data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return (self.data[index], self.target[index])


class MetaClassification(ExtendedDataset):
    """
    MetaClassification dataset given a list of functions [f_1(x), f_2(x), ... f_n(x)] that each define one class.
    """
    def __init__(self, input_dim, funs_list, num_tasks, num_way, num_shot, test_samples, spiking=False, fixed_samples=True):
        super().__init__(input_dim=input_dim, output_dim=num_way, type="classification")

        # Each task consumes `num_way` classes
        assert len(funs_list) == num_tasks * num_way

        self.funs_list = funs_list
        self.num_tasks = num_tasks
        self.num_way = num_way
        self.num_shot = num_shot
        self.test_samples = test_samples
        self.spiking = spiking
        self.fixed_samples = fixed_samples

        if fixed_samples:
            # Pregenerating the data results in fixed samples per task
            self.data_pregen = [self.sample_train_val(task) for task in range(num_tasks)]

    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        if self.fixed_samples:
            return self.data_pregen[index]
        else:
            return self.sample_train_val(index)

    def sample_task(self, index, num_samples):
        # Sample task data
        data, target = [], []
        for c in range(self.num_way):
            x, y = self.funs_list[self.num_way * index + c].sample(num_samples)
            data.append(torch.cat((y, x), dim=1))  # NOTE: switched order: y, x
            target.append(c * torch.ones(num_samples, dtype=int))

        # Shuffle task data and concatenate into tensor
        permutation = torch.randperm(num_samples * self.num_way)
        data = torch.cat(data)[permutation]
        target = torch.cat(target)[permutation]

        # Convert input from rates to spikes (assuming it is standardised in [0,1])
        if self.spiking:
            data = rates_to_latency_code(data)

        return (data, target)

    def sample_train_val(self, index):
        x_train, y_train = self.sample_task(index, self.num_shot)
        x_valid, y_valid = self.sample_task(index, self.test_samples)

        return ((x_train, y_train), (x_valid, y_valid))