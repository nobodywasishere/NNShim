sim:
	./NNShim.py -c default.toml

scripts:
	./scripts/compute_local_maxima_stats.sh
	./scripts/visualize_hotspots.sh
	./scripts/visualize_power.sh
	./scripts/plot_vs_time.sh

clean:
	rm -rf sim* nnshim_floorplan.flp

.PHONY: sim multi scripts 