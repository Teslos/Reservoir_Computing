# Redraw a Lorenz return map from the maxima CSV that plot_lorenz_map saves next
# to each figure, without rerunning the forecast.
#
# The panels are vector output from experiments that take minutes to hours, so a
# cosmetic change (a label, a title, a panel tag) used to cost a full rerun -- and
# the reruns are no longer bit-identical, which meant every tweak silently
# perturbed the published figure. Replotting from the CSV keeps the data fixed.
#
# Usage:
#   julia --project=. Reservoir-Computing-in-Julia/replot_lorenz_map.jl \
#         figures/lorenz_map_LPCTESN_quad.maxima.csv \
#         figures/lorenz_map_LPCTESN_quad.png \
#         "LPCTESN, quadratic readout (300 nodes)" "a)"

include("common.jl")
using .RCCommon
using DelimitedFiles

length(ARGS) >= 2 || error("usage: replot_lorenz_map.jl <maxima.csv> <out.png> [model] [panel]")
csv, out = ARGS[1], ARGS[2]
model = length(ARGS) >= 3 ? ARGS[3] : ""
panel = length(ARGS) >= 4 ? ARGS[4] : ""

mt, mp = let t = Float64[], p = Float64[]
    for line in Iterators.drop(eachline(csv), 1)     # drop the header
        isempty(strip(line)) && continue
        series, _, z = split(strip(line), ',')
        push!(series == "true" ? t : p, parse(Float64, z))
    end
    t, p
end
println("read $(length(mt)) true and $(length(mp)) predicted maxima from $csv")

plot_maxima_map(mt, mp, out; model = model, panel = panel)
println("wrote ", out, " (and the .pdf beside it)")
