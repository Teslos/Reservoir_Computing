# Summary chart of the FHN-reservoir topology comparison (RC_FHN_NN.jl runs,
# seed 42). Values are read from the per-topology run logs.
using CairoMakie

topologies = ["grid\n(224)", "erdos_renyi\n(414)", "barabasi_albert\n(480)",
              "watts_strogatz\n(512)", "complete\n(4032)"]
valid_lyap = [0.91, 0.87, 0.39, 0.35, 0.17]   # closed-loop valid time, Lyapunov times
open_mse = [0.166, 0.091, 0.146, 0.151, 0.031] # teacher-forced test MSE

fig = Figure(size = (1000, 450))
ax1 = Axis(fig[1, 1], title = "Closed-loop forecast skill (autonomous)",
           ylabel = "Valid prediction time (Lyapunov times)",
           xticks = (1:5, topologies), xticklabelrotation = 0.3)
barplot!(ax1, 1:5, valid_lyap, color = :firebrick)

ax2 = Axis(fig[1, 2], title = "Open-loop reconstruction error (teacher-forced)",
           ylabel = "Test MSE", xticks = (1:5, topologies),
           xticklabelrotation = 0.3)
barplot!(ax2, 1:5, open_mse, color = :steelblue)

Label(fig[0, :], "FHN reservoir (64 nodes) topology comparison - " *
                 "labels show (directed edge count)", fontsize = 16)
save("figures/FHN_topology_comparison.png", fig)
println("saved figures/FHN_topology_comparison.png")
