#!/usr/bin/env python

import os
import subprocess
from itertools import product

from typing import Dict, List
from string import Template
from pathlib import Path


benchmarks: List[str] = ["DLRM", "Googlenet"]
benchmarks_max_str_len = max([len(b) for b in benchmarks] + [len("benchmark")])
row_col_skipping: List[bool] = [True, False]
array_size: List[int] = [32, 64, 128, 256]
clock_rates: List[float] = [7e8, 1e9, 1.3e9, 1.6e9]
aspect_ratios: List[int] = [1, 2]

path_to_NNShim = Path(".")

template_fname = path_to_NNShim / "default_template.toml"

with open(template_fname) as fi:
    fi_contents: str = fi.read()
    template_str: Template = Template(fi_contents)

# for benchmark in benchmarks:
#     if os.path.isdir(f"{benchmark}_sim"):
#         shutil.rmtree(f"{benchmark}_sim", ignore_errors=True)
#     os.mkdir(f"{benchmark}_sim")

print(
    f'| {"benchmark".center(benchmarks_max_str_len)} | rowcol | arrsize | clkrate | aspect |'
)
print(f'|{"-"*(benchmarks_max_str_len+2)}|--------|---------|---------|--------|')
for (b, rowcol, arrsize, clkrate, aspect_ratio) in product(
    benchmarks, row_col_skipping, array_size, clock_rates, aspect_ratios
):
    print(
        f'| {b.center(benchmarks_max_str_len)} | {"  " if rowcol else " "}{rowcol} | {arrsize:7} | {clkrate:1.1e} | {aspect_ratio:6.2f} |'
    )
    if Path(f"{b}_sim/{arrsize}_{clkrate}_{rowcol}_{aspect_ratio}").exists():
        print("Skipping...")
        continue

    actmap_fname = f'{b}/{b}_scale-{arrsize}_{"rcs" if rowcol == True else "NA"}_10000/activityArray.npy'
    bufmap_fname = f'{b}/{b}_scale-{arrsize}_{"rcs" if rowcol == True else "NA"}_10000/bufferBandwidthArray.npy'

    file_contents = template_str.substitute(
        activity_map=actmap_fname,
        buffer_map=bufmap_fname,
        clock_rate=f"{int(clkrate):d}",
        aspect_ratio=aspect_ratio,
        row_skipping=f'{"true" if rowcol else "false"}',
        col_skipping=f'{"true" if rowcol else "false"}',
        arr_x=arrsize,
        arr_y=arrsize,
        buffer_area=f"{65004*(arrsize/32)*(arrsize/32)}",
    )
    with open("simgen.toml", "w") as fo:
        fo.write(file_contents)

    subprocess.run([f"{path_to_NNShim}/NNShim.py", "-c", "simgen.toml"])
    subprocess.run([f"{path_to_NNShim}/scripts/compute_local_maxima_stats.sh"])
    # subprocess.run([f"{path_to_NNShim}/scripts/visualize_hotspots.sh"])
    # subprocess.run([f"{path_to_NNShim}/scripts/visualize_power.sh"])

    subprocess.run(
        f"mkdir -p {b}_sim/{arrsize}_{clkrate}_{rowcol}_{aspect_ratio}".split(" ")
    )
    subprocess.run(
        f"mv sim {b}_sim/{arrsize}_{clkrate}_{rowcol}_{aspect_ratio}".split(" ")
    )
