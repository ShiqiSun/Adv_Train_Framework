import copy
import h5py
import numpy as np
import time
from pip import main
import torch.nn as nn

import utils.landscape.mpi4pytorch as mpi
import utils.landscape.eval as evaluation
import utils.landscape.scheduler as scheduler
from utils.dataset.build_datasetloader import build_dataloader
from utils.models.build_classifier import build_classifier
import utils.landscape.net_plotter as net_plotter
from utils.landscape.costume_dataset import build_custom_dataset
from utils.train.trainer import Trainer

def build_landscape(cfg, log=None):

    comm, rank, nproc = None, 0, 1

    model = build_classifier(cfg, log=log)
    # data_loader = build_dataloader(cfg, log=log)
    
    cfg.is_distributed = False
    trainer = Trainer(
        model=model,
        data_loader=None,
        cfg=cfg,
        log=log
    ) 

    data_loader = build_custom_dataset(cfg,model, log=log)
    weight = net_plotter.get_weights(model)
    state = copy.deepcopy(model.state_dict())

    dir_file = net_plotter.setup_direction(model)
    surf_file = net_plotter.setup_surface_file(dir_file)
    print(surf_file)
    mpi.barrier(comm)
    d = net_plotter.load_directions(dir_file)

    mpi.barrier(comm)
    crunch(surf_file, model, weight, 
    state, d, data_loader["test"], 'train_loss', 'train_acc', comm=comm)
    print(surf_file)
    return

def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key,
            dir_type = "weights", comm=None):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """
    f = h5py.File(surf_file, 'r+')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        f[loss_key] = losses
        f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

     # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.

    # print('Computing %d values for rank %d'% (len(inds), rank))
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)
    print('Computing %d values for rank %d'% (len(inds), 0))
    start_time = time.time()
    total_sync = 0.0
    rank = 0
    criterion = nn.CrossEntropyLoss()
    # if loss_name == 'mse':
    #     criterion = nn.MSELoss()

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
    # Load the weights corresponding to those coordinates into the net
        coord = coords[count]
        if dir_type == 'weights':
            net_plotter.set_weights(net, w, d, coord)
        elif dir_type == 'states':
            net_plotter.set_states(net, s, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        loss, acc = evaluation.eval_loss(net, criterion, dataloader, use_cuda =True)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses     = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        f[loss_key][:] = losses
        f[acc_key][:] = accuracies
        f.flush()

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))

        # This is only needed to make MPI run smoothly. If this process has less work than
        # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    # for i in range(max(inds_nums) - len(inds)):
    #     losses = mpi.reduce_max(comm, losses)
    #     accuracies = mpi.reduce_max(comm, accuracies)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (0, total_time, total_sync))
    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (0, total_time, total_sync))

    f.close()
