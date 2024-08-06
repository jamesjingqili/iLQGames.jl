
using iLQGames
import iLQGames: dx
import BenchmarkTools
using Plots
using ForwardDiff
using LinearAlgebra
using Statistics
using MAT
using Plots.PlotMeasures

plot_path = "manipulation/"
data_path = "/mnt/2312dd0d-0348-41b0-8596-b79a2cc21f6b/Dropbox/Matlab_codes/active_inference/active_teaching_LQ_convergence/"

ff = matopen(data_path*"manipulation.mat");
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

plot(thetas[1,:], mean_p1_passive[1,:], ribbon = var_p1_passive[1,:], label = "Passive", color=passive_color,linewidth=2)
plot!(thetas[1,:], mean_p1_active[1,:], ribbon = var_p1_active[1,:], label = "Active", xlabel="θ", ylabel="Regret", left_margin = 15mm, bottom_margin = 5mm, color=active_color,linewidth=2)
savefig(plot_path*"manipulation_p1_regret.png")

plot(thetas[1,:], mean_p2_passive[1,:], ribbon = var_p2_passive[1,:], label = "Passive", color = passive_color,linewidth=2)
plot!(thetas[1,:], mean_p2_active[1,:], ribbon = var_p2_active[1,:], label = "Active", xlabel="θ", ylabel="Regret", left_margin = 15mm, bottom_margin = 5mm, color=active_color,linewidth=2)
savefig(plot_path*"manipulation_p2_regret.png")










# ----------


plot_path = "manipulation/"
data_path = "/mnt/2312dd0d-0348-41b0-8596-b79a2cc21f6b/Dropbox/Matlab_codes/active_inference/active_teaching_LQ_convergence/"

ff = matopen(data_path*"manipulation_no_teaching.mat");
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

plot(thetas[1,:], mean_p1_passive[1,:], ribbon = var_p1_passive[1,:], label = "Passive", color=passive_color,linewidth=2)
plot!(thetas[1,:], mean_p1_active[1,:], ribbon = var_p1_active[1,:], label = "Active", xlabel="θ", ylabel="Regret", left_margin = 15mm, bottom_margin = 5mm, color=active_color,linewidth=2)
savefig(plot_path*"manipulation_p1_regret_no_teaching.png")

plot(thetas[1,:], mean_p2_passive[1,:], ribbon = var_p2_passive[1,:], label = "Passive", color = passive_color,linewidth=2)
plot!(thetas[1,:], mean_p2_active[1,:], ribbon = var_p2_active[1,:], label = "Active", xlabel="θ", ylabel="Regret", left_margin = 15mm, bottom_margin = 5mm, color=active_color,linewidth=2)
savefig(plot_path*"manipulation_p2_regret_no_teaching.png")

