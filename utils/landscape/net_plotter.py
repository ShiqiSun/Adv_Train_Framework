import os
from os.path import exists
import string

import h5py
import numpy as np
import torch
import utils.landscape.h5utils as h5_util
from utils.fileio.path import mkdir_or_exist

device = 0
dir_file_all = "./h5_landscape_adv_clean_02026/"

def set_weights(net, weights, directions=None, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        for (p, w, d) in zip(net.parameters(), weights, changes):
            p.data = w + torch.Tensor(d).type(type(w)).to(device)



def set_states(net, states, directions=None, step=None):
    """
        Overwrite the network's state_dict or change it along directions with a step size.
    """
    if directions is None:
        net.load_state_dict(states)
    else:
        assert step is not None, 'If direction is provided then the step must be specified as well'
        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        new_states = copy.deepcopy(states)
        assert (len(new_states) == len(changes))
        for (k, v), d in zip(new_states.items(), changes):
            d = torch.tensor(d)
            v.add_(d.type(v.type()))

        net.load_state_dict(new_states)


def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


def setup_surface_file(dir_file, 
                        surf_file:string=dir_file_all,
                        x: string= "-0.2:0.2:20",
                        y: string="-0.2:0.2:20"
                        ):
    
    # skip if the direction file already exists
    mkdir_or_exist(surf_file)
    surf_file = os.path.join(surf_file, 'surface.h5')
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return surf_file

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file
    xmin, xmax, xnum = [float(a) for a in x.split(':')]
    xnum = int(xnum)
    ymin, ymax, ynum = (None, None, None)
    if y:
        ymin, ymax, ynum = [float(a) for a in y.split(':')]
        ynum = int(ynum)
        assert ymin and ymax and ynum, \
            'You specified some arguments for the y axis, but not all'
    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(xmin, xmax, num=xnum)
    f['xcoordinates'] = xcoordinates

    if y:
        ycoordinates = np.linspace(ymin, ymax, num=ynum)
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file


def setup_direction(model,
                    dir_file:string =dir_file_all,
                    y: string="-1:1:51",
                    dir_type: string="weights",
                    xignore: string="biasbn",
                    yignore:string="biasbn",
                    xnorm: string="filter",
                    ynorm: string="filter",
                    same_dir:bool = False
                    ):
    """
        Setup the h5 file to store the directions.
        - xdirection, ydirection: The pertubation direction added to the mdoel.
            The direction is a list of tensors.
    """
    print('-------------------------------------------------------------------')
    print('setup_direction')
    print('-------------------------------------------------------------------')

    # Setup env for preventing lock on h5py file for newer h5py versions
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    mkdir_or_exist(dir_file)
    dir_file = os.path.join(dir_file, 'land.h5')
    # Skip if the direction file already exists
    if exists(dir_file):
        f = h5py.File(dir_file, 'r')
        if (y and 'ydirection' in f.keys()) or 'xdirection' in f.keys():
            f.close()
            print ("%s is already setted up" % dir_file)
            return dir_file
        f.close()

    # Create the plotting directions
    f = h5py.File(dir_file,'w') # create file, fail if exists
    print("Setting up the plotting directions...")
    xdirection = create_random_direction(model, dir_type, xignore, xnorm)
    h5_util.write_list(f, 'xdirection', xdirection)

    if y:
        if same_dir:
            ydirection = xdirection
        else:
            ydirection = create_random_direction(model, dir_type, yignore, ynorm)
        h5_util.write_list(f, 'ydirection', ydirection)

    f.close()
    print ("direction file created: %s" % dir_file)
    return dir_file

def create_random_direction(net, dir_type='weights', ignore='biasbn', norm='filter'):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
    if dir_type == 'weights':
        weights = get_weights(net) # a list of parameters.
        direction = get_random_weights(weights)
        normalize_directions_for_weights(direction, weights, norm, ignore)
    elif dir_type == 'states':
        states = net.state_dict() # a dict of parameters, including BN's running mean/var.
        direction = get_random_states(states)
        normalize_directions_for_states(direction, states, norm, ignore)

    return direction


def get_random_states(states):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    return [torch.randn(w.size()).to(device) for k, w in states.items()]


def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size()).to(device) for w in weights]


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def normalize_directions_for_states(direction, states, norm='filter', ignore='ignore'):
    assert(len(direction) == len(states))
    for d, (k, w) in zip(direction, states.items()):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def load_directions(dir_file):
    """ Load direction(s) from the direction file."""

    f = h5py.File(dir_file, 'r')
    if 'ydirection' in f.keys():  # If this is a 2D plot
        xdirection = h5_util.read_list(f, 'xdirection')
        ydirection = h5_util.read_list(f, 'ydirection')
        directions = [xdirection, ydirection]
    else:
        directions = [h5_util.read_list(f, 'xdirection')]

    return directions