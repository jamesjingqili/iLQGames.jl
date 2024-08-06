
using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff
using LinearAlgebra
using Statistics
using MAT
using Plots.PlotMeasures
plot_path = "lunar_lander/"
data_path = "/mnt/2312dd0d-0348-41b0-8596-b79a2cc21f6b/Dropbox/Matlab_codes/active_inference/active_teaching_LQ_convergence/"


ff = matopen(data_path*"lunar_lander.mat");
mean_p1_passive = read(ff,"mean_p1_passive")
var_p1_passive = read(ff,"var_p1_passive")
mean_p1_active = read(ff,"mean_p1_active")
var_p1_active = read(ff,"var_p1_active")
mean_p2_passive = read(ff,"mean_p2_passive")
var_p2_passive = read(ff,"var_p2_passive")
mean_p2_active = read(ff,"mean_p2_active")
var_p2_active = read(ff,"var_p2_active")
thetas = read(ff,"thetas")
active_color = "#ff910a"
passive_color = "#828282"
complete_color = "#1c9993"

plot(thetas[1,:], mean_p1_passive[1,:], ribbon = var_p1_passive[1,:], label = "Passive", color=passive_color, linewidth = 2)
plot!(thetas[1,:], mean_p1_active[1,:], ribbon = var_p1_active[1,:], label = "Active", xlabel="θ", ylabel="Regret",left_margin=15mm, bottom_margin=5mm, color = active_color, linewidth = 2)
savefig(plot_path*"lunar_lander_p1_regret_new.png")



plot(thetas[1,:], mean_p2_passive[1,:], ribbon = var_p2_passive[1,:], label = "Passive", color = passive_color, linewidth = 2)
plot!(thetas[1,:], mean_p2_active[1,:], ribbon = var_p2_active[1,:], label = "Active", xlabel="θ", ylabel="Regret",left_margin=15mm, bottom_margin=5mm, color=active_color, linewidth = 2)
savefig(plot_path*"lunar_lander_p2_regret_new.png")




# ----------------------------------- No teaching ----------------------------------- #



ff = matopen(data_path*"lunar_lander_no_teaching.mat");
mean_p1_passive_new = read(ff,"mean_p1_passive")
var_p1_passive_new = read(ff,"var_p1_passive")
mean_p1_active_new = read(ff,"mean_p1_active")
var_p1_active_new = read(ff,"var_p1_active")
mean_p2_passive_new = read(ff,"mean_p2_passive")
var_p2_passive_new = read(ff,"var_p2_passive")
mean_p2_active_new = read(ff,"mean_p2_active")
var_p2_active_new = read(ff,"var_p2_active")
thetas = read(ff,"thetas")
active_color = "#ff910a"
passive_color = "#828282"
complete_color = "#1c9993"
fill_alpha = 0.4
plot(thetas[1,:], mean_p1_passive_new[1,:], ribbon = var_p1_passive_new[1,:], label = "", color=passive_color, linewidth = 2, fillalpha = fill_alpha)
plot!(thetas[1,:], mean_p1_active_new[1,:], ribbon = var_p1_active_new[1,:], label = "", xlabel="", ylabel="",left_margin=2.5cm, bottom_margin=.5cm,
color = active_color, linewidth = 2, fillalpha = fill_alpha)
plot!(size=(500,320),grid=false,tickfontsize=14)
savefig(plot_path*"lunar_lander_p1_regret_no_teaching.pdf")



plot(thetas[1,:], mean_p2_passive_new[1,:], ribbon = var_p2_passive_new[1,:], label = "", color = passive_color, linewidth = 2, fillalpha = fill_alpha)
plot!(thetas[1,:], mean_p2_active_new[1,:], ribbon = var_p2_active_new[1,:], label = "", xlabel="", ylabel="",left_margin=2.5cm, bottom_margin=.5cm,
color=active_color, linewidth = 2, fillalpha = fill_alpha)
plot!(size=(500,320),grid=false,tickfontsize=14)
savefig(plot_path*"lunar_lander_p2_regret_no_teaching.pdf")








plot(thetas[1,:], mean_p1_passive[1,:], ribbon = var_p1_passive[1,:], label = "Passive", color=passive_color, linewidth = 2)
plot!(thetas[1,:], mean_p1_active[1,:], ribbon = var_p1_active[1,:], label = "Active", xlabel="θ", ylabel="Regret",left_margin=15mm, bottom_margin=5mm, color = active_color, linewidth = 2)
plot!(thetas[1,:], mean_p1_active_new[1,:], ribbon = var_p1_active_new[1,:], label = "Pure task cost", linestyle=:dash)
savefig(plot_path*"lunar_lander_p1_regret_comparison.png")



plot(thetas[1,:], mean_p2_passive[1,:], ribbon = var_p2_passive[1,:], label = "Passive", color = passive_color, linewidth = 2)
plot!(thetas[1,:], mean_p2_active[1,:], ribbon = var_p2_active[1,:], label = "Active", xlabel="θ", ylabel="Regret",left_margin=15mm, bottom_margin=5mm, color=active_color, linewidth = 2)
plot!(thetas[1,:], mean_p2_active_new[1,:], ribbon = var_p2_active_new[1,:], label = "Pure task cost", linestyle=:dash)
savefig(plot_path*"lunar_lander_p2_regret_comparison.png")

