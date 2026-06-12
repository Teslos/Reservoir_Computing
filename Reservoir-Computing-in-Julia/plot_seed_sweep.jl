# Seed-sweep comparison of grid vs Erdos-Renyi FHN reservoirs
# (RC_FHN_NN.jl, seeds 1-5, nofigs runs). Values from the RESULT lines.
using CairoMakie
using Statistics

seeds = 1:5
grid_lyap = [0.969, 0.480, 1.132, 0.426, 0.960]
er_lyap = [0.398, 0.362, 0.398, 0.326, 0.335]

fig = Figure(size = (700, 500))
ax = Axis(fig[1, 1],
          title = "FHN reservoir: closed-loop forecast skill over 5 seeds",
          ylabel = "Valid prediction time (Lyapunov times)",
          xticks = ([1, 2], ["grid (8x8)", "erdos_renyi (p=0.1)"]))
for (xpos, vals, color) in ((1, grid_lyap, :firebrick), (2, er_lyap, :steelblue))
    scatter!(ax, fill(xpos, length(vals)) .+ 0.05 .* randn(length(vals)), vals;
             color = (color, 0.7), markersize = 14, label = nothing)
    m, s = mean(vals), std(vals)
    errorbars!(ax, [xpos + 0.25], [m], [s]; whiskerwidth = 12, color = :black)
    scatter!(ax, [xpos + 0.25], [m]; color = :black, marker = :diamond,
             markersize = 12)
    text!(ax, xpos + 0.32, m; text = "$(round(m, digits = 2)) +- $(round(s, digits = 2))",
          align = (:left, :center))
end
ylims!(ax, 0, 1.3)
save("figures/FHN_seed_sweep_grid_vs_ER.png", fig)
println("saved figures/FHN_seed_sweep_grid_vs_ER.png")
