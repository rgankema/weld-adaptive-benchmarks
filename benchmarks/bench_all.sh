declare -a benches=("bloomfilter" "branch_vs_predicate" "filter_map_reorder" "join_dm" "join_dm_simple" "join_gm" "join_dm_vs_gm" "sf_overhead" "sf_overhead_instr")

for bench in "${benches[@]}"
do
    cd $bench
    python bench.py -c conf.json -o results.csv
done;