#!/usr/bin/env python

import os, sys
import argparse
from typing import Any, Dict, List, Tuple
from pandas import DataFrame # type: ignore
import toml
import logging as log

import numpy as np
import numpy.typing as npt
from math import floor, ceil, sqrt

import matplotlib.pyplot as plt # type: ignore
import pint # type: ignore
ureg = pint.UnitRegistry()

from pprint import pprint

from HotGauge.utils.floorplan import Floorplan, FloorplanElement # type: ignore
from HotGauge.thermal.analysis import local_max_stats_df, local_max_stats_from_file, local_max_stats_to_file # type: ignore
from HotGauge.thermal.ICE import ICETransientSimRAW, get_stk_template, ICETransientSim, load_3DICE_grid_file # type: ignore
from HotGauge.thermal.metrics import severity_metric # type: ignore
from HotGauge.thermal.utils import C_to_K # type: ignore
from HotGauge.visualization.ICE_plt import find_stk_file, plot_stats, plot_dist # type: ignore


debug = False

def rowSkipping(act_map: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """Generates a new activity map with twice the number of rows. Each row has on average half the activity.

    Args:
        act_map (npt.NDArray[np.uint16]): Input activity map.

    Returns:
        npt.NDArray[np.uint16]: Activity map with twice the number of rows. Sum of activity per sample should still be the same.
    """
    samples, rows, cols = act_map.shape
    new_rows = rows*2
    new_actmap: npt.NDArray[np.uint16] = np.zeros((samples, new_rows, cols), int)
    for s in range(samples):
        for r in range(new_rows):
            for c in range(cols):
                expand_from = act_map[s, floor(r/2), c]
                if (r+s)%2:
                    new_actmap[s, r, c] = floor(expand_from/2)
                else:
                    new_actmap[s, r, c] = ceil(expand_from/2)
    return new_actmap

def colSkipping(act_map: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """Generates a new activity map with twice the number of cols. Each col has on average half the activity.

    Args:
        act_map (npt.NDArray[np.uint16]): Input activity map.
        
    Returns:
        npt.NDArray[np.uint16]: Activity map with twice the number of cols. Sum of activity per sample should still be the same.
    """
    samples, rows, cols = act_map.shape
    new_cols = cols*2
    new_actmap: npt.NDArray[np.uint16] = np.zeros((samples, rows, new_cols), int)
    for s in range(samples):
        for r in range(rows):
            for c in range(new_cols):
                expand_from = act_map[s, r, floor(c/2)]
                if (c+floor(s/2))%2:
                    new_actmap[s, r, c] = floor(expand_from/2)
                else:
                    new_actmap[s, r, c] = ceil(expand_from/2)
    return new_actmap

def pePowerMap(config: Dict[str, Any], bucket_size_cycles: int, time_slot: float) -> npt.NDArray[np.uint16]:
    """Generate power map for PE units from activity map file

    Args:
        config (dict): A hash of the config file passed to NNShim
        bucket_size_cycles (int): Number of cycles per bucket
        time_slot (float): Number of seconds per bucket

    Returns:
        dict: A power map from the activity array
    """
    # load activity map from file
    amap_fpath = os.path.abspath(config['scalesim']['activity_map_file']) 
    amap_repetition = config['scalesim']['neural_network_repetition']
    amap = np.load(amap_fpath, allow_pickle=True)
    
    #apply row and column skipping
    if config['systolic_array']['row_skipping'] == True:
        amap = rowSkipping(amap)
    
    #apply row and column skipping
    if config['systolic_array']['col_skipping'] == True:
        amap = colSkipping(amap)
    
    amap = np.tile(amap, (amap_repetition, 1, 1))

    # get the energy per operation
    pe_per_op_energy_J = config['processing_element']['per_op_energy_J']

    # get the size of the activity map and the total number of timesteps
    amap_cycles, amap_x, amap_y = amap.shape
    num_timesteps = ceil(amap_cycles / bucket_size_cycles)

    # sum up activity map cycles by bucket size
    power_map: npt.NDArray[np.uint16] = np.zeros( (num_timesteps, amap_x, amap_y), dtype=float)
    for idx, val in enumerate(power_map):
        start_cycle = (idx)*bucket_size_cycles
        stop_cycle  = min( (idx+1)*bucket_size_cycles, amap_cycles )
        amap_region = amap[start_cycle : stop_cycle]
        power_map[idx] = np.sum(amap_region, axis=0)

    # scale to the power usage per activity
    power_map *= pe_per_op_energy_J / time_slot

    # round stuff off for 3d-ice
    for i in range(len(power_map)):
        for j in range(len(power_map[i])):
            for k in range(len(power_map[i][j])):
                power_map[i][j][k] = int(power_map[i][j][k] * 10**8) / 10**8

    return power_map


def bufferPowerMap(config: Dict[str, Any], bucket_size_cycles: int, time_slot: float) -> Dict[str, npt.NDArray[np.uint16]]:
    """Generate power map for the input, filter, and output buffers from activity map file

    Args:
        config (dict): A hash of the config file passed to NNShim
        bucket_size_cycles (int): Number of cycles per bucket
        time_slot (float): Number of seconds per bucket

    Returns:
        dict: 3 power maps for each of the buffers
    """

    # load buffer activity map from file
    bmap_fpath = os.path.abspath(config['scalesim']['buffer_bandwidth_file']) 
    bmap_repetition = config['scalesim']['neural_network_repetition']
    bmap = np.load(bmap_fpath)

    bmap = np.tile(bmap, (1, bmap_repetition))
   

    # load the read/write energies
    buf_read_energy_J = config['buffer']['read_energy_J']
    buf_write_energy_J = config['buffer']['write_energy_J']

    # get the size of the array and the total number of timesteps
    buffer_bw_buffer_idx, buffer_total_cycles = bmap.shape
    num_timesteps = ceil( buffer_total_cycles / bucket_size_cycles )

    # sum up activity map cycles by bucket size for each buffer
    power_map: npt.NDArray[np.uint16] = np.zeros( (num_timesteps, buffer_bw_buffer_idx), dtype=float)
    for idx, val in enumerate(power_map):
        start_cycle = (idx)*bucket_size_cycles
        stop_cycle = min( (idx+1)*bucket_size_cycles, buffer_total_cycles )
        bmap_region = bmap[:, start_cycle : stop_cycle]
        power_map[idx, 0] = np.sum(bmap_region[0, :])
        power_map[idx, 1] = np.sum(bmap_region[1, :])
        power_map[idx, 2] = np.sum(bmap_region[2, :])

    # scale to the power usage per activity
    power_dict = {}
    power_dict['input']  = power_map[:, 0] * buf_read_energy_J / time_slot
    power_dict['filter'] = power_map[:, 1] * buf_read_energy_J / time_slot
    power_dict['output'] = power_map[:, 2] * buf_write_energy_J / time_slot

    # round stuff off for 3d-ice
    for i in power_dict.keys():
        for j in range(len(power_dict[i])):
            power_dict[i][j] = int(power_dict[i][j] * 10**8) / 10**8

    return power_dict


def peFloorplan(config: Dict[str, Any], power_map: npt.NDArray[np.uint16], start_x: float, start_y: float) -> Tuple[List[FloorplanElement], Dict[str, List[int]]]:
    """Layout the PE units into a floorplan

    Args:
        config (dict): A hash of the config file passed to NNShim
        power_map (list): A power map from the activity array
        start_x (float): Starting x posiiton for the array
        start_y (float): Starting y posiiton for the array

    Returns:
        list: list of elements, dict of element powers
    """
    # get array size
    pe_size_array: Dict[str, int] = config['processing_element']['array_size']

    # get PE size
    pe_size_um: Dict[str, int] = config['processing_element']['element_size']

    # get power trace array size
    pe_size_power: Dict[str, int] = config['processing_element']['power_array_size']
    pe_power_um = {
        'x': pe_size_um['x']*int(pe_size_array['x']/pe_size_power['x']),
        'y': pe_size_um['y']*int(pe_size_array['y']/pe_size_power['y'])
    }

    # get spacing between elements
    pe_spacing_um: int = config['processing_element']['pe_spacing_um']

    # get the aspect ratio and reshape the power map
    if 'aspect_ratio' in config['systolic_array']:
        aspect_ratio: int = config['systolic_array']['aspect_ratio']
        aspected_size_x: int = int(pe_size_power['x']/aspect_ratio)
        aspected_size_y: int = int(pe_size_power['y']*aspect_ratio)
        num_cycles: int = power_map.shape[0]
        power_map = np.reshape(power_map, 
            (
                num_cycles, 
                int(pe_size_power['x']/aspect_ratio), 
                int(pe_size_power['y']*aspect_ratio)
            )
        )
    else:
        aspect_ratio = 1
        aspected_size_x = pe_size_power['x']
        aspected_size_y = pe_size_power['y']


    elements: List[FloorplanElement] = []
    element_powers: Dict[str, List[int]] = {}

    # add each element trace to a list and its power information to a power dictionary
    for x in range(aspected_size_x):
        for y in range(aspected_size_y):
            element_name = f'PE{x:04d}{y:04d}'
            elements.append(
                FloorplanElement(
                    element_name, 
                    pe_power_um['x'],
                    pe_power_um['y'],
                    (pe_power_um['x'] + pe_spacing_um) * x + start_x,
                    (pe_power_um['x'] + pe_spacing_um) * y + start_y
                )
            )

            element_powers[element_name] = list(power_map[:, x, y])

    return elements, element_powers


def bufferFloorplan(config: Dict[str, Any], flp: Floorplan) -> None:
    """Places the input, filter, and output buffers on an HotGauge Floorplan

    Args:
        config (dict): A hash of the config file passed to NNShim
        flp (Floorplan): Floorplan object for NNShim to place buffers on
    """

    flp.auto_place_element('BUF0', config['buffer']['area'], 'above')
    flp.auto_place_element('BUF1', config['buffer']['area'], 'left')
    flp.auto_place_element('BUF2', config['buffer']['area'], 'right')


def simulate(config: Dict[str, Any], element_powers: Dict[str, List[int]], flp: Floorplan, time_slot: float) -> None:
    """Simulate a systollic array with HotGauge

    Args:
        config (dict): A hash of the config file passed to NNShim
        element_powers (dict): All of the power maps for buffers and PE units
        flp (Floorplan): HotGauge Floorplan to use for simulation
        time_slot (float): Time step size in seconds
    """
    # TODO: this is a temp fix, figure out what to use for final heatsink model
    stk_template: str = get_stk_template(os.path.dirname(os.path.abspath(__file__)) + '/' + config['3d_ice']['stk_template']) 

    # create/generate floorplan with elements
    # TODO: Figure out difference between 'hotspot' and '3D-ICE' frmt
    flp_file: str = os.path.dirname(os.path.abspath(__file__)) + '/' + config['3d_ice']['floorplan_file']
    flp.to_file(flp_file, element_powers=element_powers)

    # get techology node from config
    tech_node: str = config['3d_ice']['tech_node']

    # get output directory for 3d-ice from config
    odir: str = os.path.dirname(os.path.abspath(__file__)) + '/' + config['3d_ice']['output_directory']
    
    # get initial temperature from config
    initial_t: int = C_to_K(config['3d_ice']['initial_t'])

    # set simulation outputs
    sim_outputs = [
        ICETransientSimRAW.OUTPUT_TSTACK_FINAL,
        ICETransientSimRAW.DIE_TMAP_OUTPUT,
        ICETransientSimRAW.DIE_PMAP_OUTPUT,
        ICETransientSimRAW.DIE_TFLP_OUTPUT
    ]
    
    # get heatsink arguments from config
    heatsink_args: str = config['3d_ice']['heatsink_args']

    # initialize simulation object
    sim = ICETransientSimRAW(
        time_slot, stk_template, flp_file, [], tech_node, odir, 
        initial_temp=initial_t, output_list=sim_outputs, plugin_args=heatsink_args
        # time_slot, stk_template, flp_template, ptraces, tech_node, sim_dir
    )

    # run hotgauge simulation
    ICETransientSimRAW.run([sim])

def calc_severity(output_dir: str) -> DataFrame:
    df = local_max_stats_df(f"{output_dir}/die_grid.temps", mltd_radius_px=20)
    local_max_stats_to_file(df, f"{output_dir}/die_grid.temps.2dmaxima")
    df = local_max_stats_from_file(f"{output_dir}/die_grid.temps.2dmaxima")
    return severity_metric(df['MLTD'], df['temp_xy'])

def plot_temp(input_file: str = "die_grid.pows", plot_type: str = "stats", output_file: str = "temp.png") -> None:
    threshold = [110]
    min_val = 25
    max_val = 135

    data = load_3DICE_grid_file(input_file, convert_K_to_C=True)

    stk_file = find_stk_file(os.path.dirname(input_file))
    if stk_file is not None:
        _, slot = ICETransientSim.get_step_slot(stk_file)
        cell_size = ICETransientSim.get_cell_size(stk_file)
    else:
        slot = None
        cell_size = None
    data_label = 'Temperature'
    unit = ureg['degree_Celsius'].units
    factor = 1
    data_label = '{}({:~P})'.format(data_label, unit)
    if plot_type == 'stats':
        plot_stats(data*factor, time_slot=slot, thresholds=threshold)
        dmin, dmax = plt.ylim()
        min_val = dmin if min_val is None else min_val
        max_val = dmax if max_val is None else max_val
        plt.ylim(min_val, max_val)
    elif plot_type == 'dist':
        plot_dist(data*factor,
                  data_label=data_label)
    else:
        raise ValueError('Invalid plot_type: {}'.format(plot_type))
    
    plt.savefig(output_file)

def main(config: Dict[str, Any]) -> None:
    sa_clock_rate: int = config['systolic_array']['clock_rate']

    cycles_per_sample: int = config['scalesim']['cycles_per_sample']
    target_timestep_min: int = config['systolic_array']['target_timestep_min']
    # cyc/sec * sec * smp/cyc = smp/bkt
    bucket_size_cycles: int = int(ceil(sa_clock_rate * target_timestep_min / cycles_per_sample))
    # smp/bkt * sec/cyc * cyc/smp = sec/bkt
    time_slot: float = bucket_size_cycles * cycles_per_sample / sa_clock_rate
    if debug:
        print(f'{cycles_per_sample=}')
        print(f'{target_timestep_min=}')
        print(f'{bucket_size_cycles=}')
        print(f'{time_slot=}')

    power_map: npt.NDArray[np.uint16] = pePowerMap(config, bucket_size_cycles, time_slot)
    buffer_map: Dict[str, npt.NDArray[np.uint16]] = bufferPowerMap(config, bucket_size_cycles, time_slot)
    
    elements, element_powers = peFloorplan(config, power_map, 0, 0)

    element_powers['BUF0'] = list(buffer_map['input'])
    element_powers['BUF1'] = list(buffer_map['filter'])
    element_powers['BUF2'] = list(buffer_map['output'])

    flp = Floorplan(elements, frmt='3D-ICE')

    bufferFloorplan(config, flp)

    flp.reset_to_origin()

    simulate(config, element_powers, flp, time_slot)

    odir: str = os.path.dirname(os.path.abspath(__file__)) + '/' + config['3d_ice']['output_directory']
    # print("severity: \n", calc_severity(odir))
    # plot_temp(f"{odir}/die_grid.pows", "stats", "temp_stats.png")

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Integrate SCALE-Sim simulations into HotGauge')
    parser.add_argument('-c', '--config', 
        help='Configuration file for NNShim', required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        debug = True

    try:
        config: Dict[str, Any] = toml.load(args.config)
    except FileNotFoundError:
        log.error(f'File not found: {args.config}')
        exit(1)

    main(config)