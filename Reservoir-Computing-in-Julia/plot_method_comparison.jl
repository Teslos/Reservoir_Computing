# Three-method comparison of closed-loop Lorenz-63 forecast skill.
# Valid prediction time (Lyapunov times) over seeds 1-5, from the RESULT
# lines of RC.jl (ESN+ridge), RC_LPCTESN.jl (continuous-time ESN), and
# RC_FHN_NN.jl (FHN physical reservoir, grid, near-bifurcation, NN readout).
using CairoMakie
using Statistics

methods = ["ESN + ridge\n(discrete, NR=500)",
           "LPCTESN\n(continuous, NR=300)",
           "LPCTESN quad\n(continuous, [r; r²])",
           "FHN physical\n(grid, NR=256)"]
data = [[6.493, 9.699, 6.448, 9.183, 8.413],   # ESN + ridge
        [1.259, 0.842, 0.987, 1.014, 0.942],   # LPCTESN linear
        [1.041, 1.105, 1.032, 1.050, 1.041],   # LPCTESN quadratic readout
        [0.389, 0.996, 1.078, 0.408, 0.480]]   # FHN (NN readout)

fig = Figure(size = (920, 540))
ax = Axis(fig[1, 1], title = "Closed-loop Lorenz-63 forecast skill (seeds 1-5)",
          ylabel = "Valid prediction time (Lyapunov times)",
          xticks = (1:4, methods))
for (i, vals) in enumerate(data)
    m, s = mean(vals), std(vals)
    barplot!(ax, [i], [m], color = (:steelblue, 0.4), width = 0.5)
    errorbars!(ax, [i], [m], [s]; whiskerwidth = 14, color = :black)
    scatter!(ax, fill(i, length(vals)) .+ 0.16 .* randn(length(vals)), vals;
             color = :firebrick, markersize = 11)
    text!(ax, i, m + s + 0.4; text = "$(round(m, digits=2)) ± $(round(s, digits=2))",
          align = (:center, :bottom), fontsize = 14)
end
ylims!(ax, 0, 12)
save("figures/method_comparison.png", fig)
println("saved figures/method_comparison.png")
for (n, vals) in zip(methods, data)
    println(replace(n, "\n" => " "), ": ", round(mean(vals), digits=2),
            " ± ", round(std(vals), digits=2), " LT")
end
