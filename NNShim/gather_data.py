#!/usr/bin/env python

import csv
from pathlib import Path

from numpy import average

sim_folder = input("Sim folder: ")
benchmark = sim_folder.split("/")[-1].split("_")[0]

print(f"| clkrate | aspect | rowcol | arrsize | max temp | max severity |")
print(f"|---------|--------|--------|---------|----------|--------------|")

for dir in Path(sim_folder).iterdir():
    arrsize, clkrate, rowcol, aspect = dir.name.split("_")
    with open(dir / "sim" / "die_grid.temps.2dmaxima.csv", newline="") as csvfile:
        data = list(csv.DictReader(csvfile))
        temps = [float(t["temp_xy"]) for t in data]
        sevs = [float(t["pos_MLTD"]) for t in data]
        max_temp = max(temps)
        max_sev = max(sevs)
    print(
        f'| {float(clkrate)/1000000000:1.1f}     |      {int(aspect)} |  {" " if rowcol == "True" else ""}{rowcol} | {arrsize:7} | {max_temp:3.05f} | {max_sev:1.05f}      |'
    )

# https://tableconvert.com/markdown-to-csv
