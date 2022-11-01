using Plots
using JLD2
using iLQGames
using Statistics

# the other plots are done in iB_dubins_GD_x0.jl

# 3 cars---------------------
t1=load("Walnut_Creek_Indianna_var_KKT_clean_3cars_partial")
t2=load("Walnut_Creek_Indianna_var_KKT_clean_3cars_full")
t3=load("Walnut_Creek_Indy_baobei_GD_3car_partial_x0")
t4=load("Walnut_Creek_Indy_baobei_GD_3car_full_x0")

noise_level_list = t1["noise_level_list"]
num_noise_level = length(noise_level_list)
# partial
fillalpha=0.5
# tmp1 = [mean(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp2 = [mean(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp3 = [mean(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp1_var = [var(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp2_var = [var(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp3_var = [var(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp14, tmp15, tmp16, tmp14_var, tmp15_var, tmp16_var = t3["tmp14"], t3["tmp15"], t3["tmp16"], t3["tmp14_var"], t3["tmp15_var"], t3["tmp16_var"]


num_obs=length(t1["inv_sol_list"][1])
for noise in 1:num_noise_level
	tmp1[noise] = mean(log(t1["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2[noise] = mean(log(t1["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3[noise] = mean(log(t1["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp1_var[noise] = var(log(t1["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2_var[noise] = var(log(t1["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3_var[noise] = var(log(t1["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp14[noise] = mean(log(t3["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15[noise] = mean(log(t3["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16[noise] = mean(log(t3["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp14_var[noise] = var(log(t3["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15_var[noise] = var(log(t3["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16_var[noise] = var(log(t3["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
end


plt1=plot()
plt1=plot!(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1,fillalpha=fillalpha, line=:dash,linewidth=3, color="red", label = "", xlabel="noise variance",title="Partial obs. and incomplete traj.")
plt1=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="blue", label = "")
plt1=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="orange", label = "")
plt1=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1,fillalpha=fillalpha, color="red",linewidth=3, label="")
plt1=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1,fillalpha=fillalpha, color="blue",linewidth=3, label = "")
plt1=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1,fillalpha=fillalpha, color="orange",linewidth=3, label="", ylabel = "value in log scale")

# full
# tmp1 = [mean(t2["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp2 = [mean(t2["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp3 = [mean(t2["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp1_var = [var(t2["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp2_var = [var(t2["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp3_var = [var(t2["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
# tmp14, tmp15, tmp16, tmp14_var, tmp15_var, tmp16_var = t4["tmp14"], t4["tmp15"], t4["tmp16"], t4["tmp14_var"], t4["tmp15_var"], t4["tmp16_var"]
num_obs=length(t2["inv_sol_list"][1])
for noise in 1:num_noise_level
	tmp1[noise] = mean(log(t2["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2[noise] = mean(log(t2["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3[noise] = mean(log(t2["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp1_var[noise] = var(log(t2["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2_var[noise] = var(log(t2["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3_var[noise] = var(log(t2["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp14[noise] = mean(log(t4["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15[noise] = mean(log(t4["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16[noise] = mean(log(t4["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp14_var[noise] = var(log(t4["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15_var[noise] = var(log(t4["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16_var[noise] = var(log(t4["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
end
plt2=plot()
plt2=plot!(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1,fillalpha=fillalpha, line=:dash,linewidth=3, color="red", label = "Baseline, distance to noisy observation data", xlabel="noise variance",legend = :topright,title="Full obs. and complete traj.")
plt2=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="blue", label = "Baseline, distance to ground truth data")
plt2=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="orange", label = "Baseline, generalization loss")
plt2=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1,fillalpha=fillalpha, color="red",linewidth=3, label="Algorithm 1, distance to noisy observation data")
plt2=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1,fillalpha=fillalpha, color="blue",linewidth=3, label = "Algorithm 1, distance to ground truth data")
plt2=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1,fillalpha=fillalpha, color="orange",linewidth=3, label="Algorithm 1, generalization loss", ylabel="value in log scale")

fullplt=plot(plt1, plt2, layout=(1,2), size = (800,400),left_margin=5Plots.mm)
display(fullplt)
savefig("cars3_comparison.pdf")

# 2 cars--------------------

t1=load("1023_baobei_KKT_x0_partial_10_corrected_state_list")
t2=load("1023_baobei_KKT_x0_full_10_corrected_state_list")
t3=load("1023_baobei_GD_2car_partial_x0")
t4=load("1023_baobei_GD_2car_full_x0")

noise_level_list = t1["noise_level_list"]
num_noise_level = length(noise_level_list)
# partial
fillalpha=0.5
tmp1 = [mean(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2 = [mean(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3 = [mean(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp1_var = [var(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2_var = [var(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3_var = [var(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp14, tmp15, tmp16, tmp14_var, tmp15_var, tmp16_var = t3["tmp14"], t3["tmp15"], t3["tmp16"], t3["tmp14_var"], t3["tmp15_var"], t3["tmp16_var"]
plt1=plot()
plt1=plot(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1,fillalpha=fillalpha, line=:dash,linewidth=3, color="red", label = "", xlabel="noise variance", legend = :topleft, ylim=(0,8.5),title="Partial obs. and incomplete traj.")
plt1=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="blue", label = "")
plt1=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="orange", label = "")
plt1=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1,fillalpha=fillalpha, color="red",linewidth=3, label="")
plt1=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1,fillalpha=fillalpha, color="blue",linewidth=3, label = "")
plt1=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1,fillalpha=fillalpha, color="orange",linewidth=3, label="",ylabel="value")

# full
tmp1 = [mean(t2["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2 = [mean(t2["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3 = [mean(t2["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp1_var = [var(t2["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2_var = [var(t2["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3_var = [var(t2["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp14, tmp15, tmp16, tmp14_var, tmp15_var, tmp16_var = t4["tmp14"], t4["tmp15"], t4["tmp16"], t4["tmp14_var"], t4["tmp15_var"], t4["tmp16_var"]
plt2=plot()
plt2=plot(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1,fillalpha=fillalpha, line=:dash,linewidth=3, color="red", label = "Baseline, distance to noisy observation data", xlabel="noise variance",legend = :topright, ylim=(0,8.5), title="Full obs. and complete traj.")
plt2=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="blue", label = "Baseline, distance to ground truth data")
plt2=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="orange", label = "Baseline, generalization loss")
plt2=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1,fillalpha=fillalpha, color="red",linewidth=3, label="Algorithm 1, distance to noisy observation data")
plt2=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1,fillalpha=fillalpha, color="blue",linewidth=3, label = "Algorithm 1, distance to ground truth data")
plt2=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1,fillalpha=fillalpha, color="orange",linewidth=3, label="Algorithm 1, generalization loss",ylabel="value")

fullplt=plot(plt1, plt2, layout=(1,2), size = (800,400),left_margin=5Plots.mm)
display(fullplt)

savefig("cars2_comparison.pdf")


# ---------------
# 2cars new
t1=load("1023_baobei_KKT_x0_partial_10_corrected_state_list")
t2=load("1023_baobei_KKT_x0_full_10_corrected_state_list")
t3=load("1023_baobei_GD_2car_partial_x0")
t4=load("1023_baobei_GD_2car_full_x0")
noise_level_list = t1["noise_level_list"]
num_noise_level = length(noise_level_list)
# partial
fillalpha=0.5

tmp1 = [mean(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2 = [mean(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3 = [mean(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp1_var = [var(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2_var = [var(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3_var = [var(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp14, tmp15, tmp16, tmp14_var, tmp15_var, tmp16_var = t3["tmp14"], t3["tmp15"], t3["tmp16"], t3["tmp14_var"], t3["tmp15_var"], t3["tmp16_var"]

num_obs=length(t1["inv_sol_list"][1])
for noise in 1:num_noise_level
	tmp1[noise] = mean(log(t1["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2[noise] = mean(log(t1["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3[noise] = mean(log(t1["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp1_var[noise] = 1/sqrt(10)*std(log(t1["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2_var[noise] = 1/sqrt(10)*std(log(t1["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3_var[noise] = 1/sqrt(10)*std(log(t1["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp14[noise] = mean(log(t3["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15[noise] = mean(log(t3["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16[noise] = mean(log(t3["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp14_var[noise] = 1/sqrt(10)*std(log(t3["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15_var[noise] = 1/sqrt(10)*std(log(t3["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16_var[noise] = 1/sqrt(10)*std(log(t3["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
end
plt1=plot()
plt1=plot!(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1,fillalpha=fillalpha, line=:dash,linewidth=3, color="red", label = "", xlabel="noise variance",title="Partial obs. and incomplete traj.")
plt1=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="blue", label = "")
plt1=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="orange", label = "")
plt1=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1,fillalpha=fillalpha, color="red",linewidth=3, label="")
plt1=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1,fillalpha=fillalpha, color="blue",linewidth=3, label = "")
plt1=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1,fillalpha=fillalpha, color="orange",linewidth=3, label="", ylabel = "distance in log scale",ylims=(-6,5))
num_obs=length(t2["inv_sol_list"][1])
for noise in 1:num_noise_level
	tmp1[noise] = mean(log(t2["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2[noise] = mean(log(t2["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3[noise] = mean(log(t2["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp1_var[noise] = 1/sqrt(10)*std(log(t2["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2_var[noise] = 1/sqrt(10)*std(log(t2["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3_var[noise] = 1/sqrt(10)*std(log(t2["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp14[noise] = mean(log(t4["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15[noise] = mean(log(t4["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16[noise] = mean(log(t4["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp14_var[noise] = 1/sqrt(10)*std(log(t4["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15_var[noise] = 1/sqrt(10)*std(log(t4["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16_var[noise] = 1/sqrt(10)*std(log(t4["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
end
plt2=plot()
plt2=plot!(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1,fillalpha=fillalpha, line=:dash,linewidth=3, color="red", label = "Baseline, to noisy observation data", xlabel="noise variance",legend = :topright,title="Full obs. and complete traj.")
plt2=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="blue", label = "Baseline, to ground truth data")
plt2=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="orange", label = "Baseline, to expert trajectory from unseen initial state")
plt2=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1,fillalpha=fillalpha, color="red",linewidth=3, label="Algorithm 1, to noisy observation data")
plt2=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1,fillalpha=fillalpha, color="blue",linewidth=3, label = "Algorithm 1, to ground truth data")
plt2=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1,fillalpha=fillalpha, color="orange",linewidth=3, label="Algorithm 1, to expert trajectory from unseen initial state", ylabel="distance in log scale", ylims=(-6,5))

fullplt=plot(plt1, plt2, layout=(1,2), size = (1000,400),left_margin=5Plots.mm,bottom_margin=5Plots.mm)
display(fullplt)
savefig("cars2_comparison_log.pdf")



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#3cars new
t1=load("Walnut_Creek_Indianna_var_KKT_clean_3cars_partial")
t2=load("Walnut_Creek_Indianna_var_KKT_clean_3cars_full")
t3=load("Walnut_Creek_Indy_baobei_GD_3car_partial_x0")
t4=load("Walnut_Creek_Indy_baobei_GD_3car_full_x0")

noise_level_list = t1["noise_level_list"]
num_noise_level = length(noise_level_list)
# partial
fillalpha=0.5
tmp1 = [mean(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2 = [mean(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3 = [mean(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp1_var = [1/sqrt(6)*std(t1["inv_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp2_var = [1/sqrt(6)*std(t1["inv_ground_truth_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp3_var = [1/sqrt(6)*std(t1["inv_mean_generalization_loss_list"][ii])[1] for ii in 1:num_noise_level]
tmp14, tmp15, tmp16, tmp14_var, tmp15_var, tmp16_var = t3["tmp14"], t3["tmp15"], t3["tmp16"], t3["tmp14_var"], t3["tmp15_var"], t3["tmp16_var"]

num_obs=length(t1["inv_sol_list"][1])
for noise in 1:num_noise_level
	tmp1[noise] = mean(log(t1["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2[noise] = mean(log(t1["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3[noise] = mean(log(t1["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp1_var[noise] = 1/sqrt(6)*std(log(t1["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2_var[noise] = 1/sqrt(6)*std(log(t1["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3_var[noise] = 1/sqrt(6)*std(log(t1["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp14[noise] = mean(log(t3["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15[noise] = mean(log(t3["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16[noise] = mean(log(t3["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp14_var[noise] = 1/sqrt(6)*std(log(t3["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15_var[noise] = 1/sqrt(6)*std(log(t3["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16_var[noise] = 1/sqrt(6)*std(log(t3["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
end
plt1=plot()
plt1=plot!(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1,fillalpha=fillalpha, line=:dash,linewidth=3, color="red", label = "", xlabel="noise variance",title="Partial obs. and incomplete traj.")
plt1=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="blue", label = "")
plt1=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="orange", label = "")
plt1=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1,fillalpha=fillalpha, color="red",linewidth=3, label="")
plt1=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1,fillalpha=fillalpha, color="blue",linewidth=3, label = "")
plt1=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1,fillalpha=fillalpha, color="orange",linewidth=3, label="", ylabel = "distance in log scale",ylims=(-6,8))
num_obs=length(t2["inv_sol_list"][1])
for noise in 1:num_noise_level
	tmp1[noise] = mean(log(t2["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2[noise] = mean(log(t2["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3[noise] = mean(log(t2["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp1_var[noise] = 1/sqrt(6)*std(log(t2["inv_loss_list"][noise][ii][1]) for ii in 1:num_obs )
	tmp2_var[noise] = 1/sqrt(6)*std(log(t2["inv_ground_truth_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp3_var[noise] = 1/sqrt(6)*std(log(t2["inv_mean_generalization_loss_list"][noise][ii][1]) for ii in 1:num_obs)
	tmp14[noise] = mean(log(t4["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15[noise] = mean(log(t4["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16[noise] = mean(log(t4["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp14_var[noise] = 1/sqrt(6)*std(log(t4["optim_loss_list_list"][1][noise][1][ii]) for ii in 1:num_obs )
	tmp15_var[noise] = 1/sqrt(6)*std(log(t4["list_ground_truth_loss"][1][noise][1][ii]) for ii in 1:num_obs)
	tmp16_var[noise] = 1/sqrt(6)*std(log(t4["list_generalization_loss"][1][noise][1][ii]) for ii in 1:num_obs)
end
plt2=plot()
plt2=plot!(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1,fillalpha=fillalpha, line=:dash,linewidth=3, color="red", label = "Baseline, to noisy observation data", xlabel="noise variance",legend = :topright,title="Full obs. and complete traj.")
plt2=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="blue", label = "Baseline, to ground truth data")
plt2=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="orange", label = "Baseline, to expert trajectory from unseen initial state")
plt2=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1,fillalpha=fillalpha, color="red",linewidth=3, label="Algorithm 1, to noisy observation data")
plt2=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1,fillalpha=fillalpha, color="blue",linewidth=3, label = "Algorithm 1, to ground truth data")
plt2=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1,fillalpha=fillalpha, color="orange",linewidth=3, label="Algorithm 1, to expert trajectory from unseen initial state", ylabel="distance in log scale", ylims=(-6,8))

fullplt=plot(plt1, plt2, layout=(1,2), size = (1000,400),left_margin=5Plots.mm,bottom_margin=5Plots.mm)
display(fullplt)
savefig("cars3_comparison_log.pdf")





# ---------------------------- last minute
t1=load("Berkeley_var_KKT_clean_3cars_partial")
t2=load("Berkeley_var_KKT_clean_3cars_full")
t3=load("Berkeley_baobei_GD_3car_partial")
t4=load("Berkeley_baobei_GD_3car_full")
noise_level_list = t1["noise_level_list"]
num_noise_level = length(noise_level_list)
# partial
fillalpha=0.5
tmp3, tmp1, tmp2, tmp3_var,tmp1_var,tmp2_var = t1["mean1"], t1["mean2"], t1["mean3"], t1["var1"], t1["var2"], t1["var3"]
tmp14, tmp15, tmp16, tmp14_var, tmp15_var, tmp16_var = t3["tmp14"], t3["tmp15"], t3["tmp16"], t3["tmp14_var"], t3["tmp15_var"], t3["tmp16_var"]
num_obs=length(t1["inv_sol_list"][1])
plt1=plot()
plt1=plot!(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1,fillalpha=fillalpha, line=:dash,linewidth=3, color="red", label = "", xlabel="noise variance",title="Partial obs. and incomplete traj.")
plt1=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="blue", label = "")
plt1=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="orange", label = "")
plt1=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1,fillalpha=fillalpha, color="red",linewidth=3, label="")
plt1=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1,fillalpha=fillalpha, color="blue",linewidth=3, label = "")
plt1=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1,fillalpha=fillalpha, color="orange",linewidth=3, label="", ylabel = "distance in log scale",ylims=(-6,8))
num_obs=length(t2["inv_sol_list"][1])
tmp3, tmp1, tmp2, tmp3_var,tmp1_var,tmp2_var = t2["mean1"], t2["mean2"], t2["mean3"], t2["var1"], t2["var2"], t2["var3"]
tmp14, tmp15, tmp16, tmp14_var, tmp15_var, tmp16_var = t4["tmp14"], t4["tmp15"], t4["tmp16"], t4["tmp14_var"], t4["tmp15_var"], t4["tmp16_var"]

plt2=plot()
plt2=plot!(noise_level_list, tmp1, ribbons=(tmp1_var, tmp1_var),alpha=1,fillalpha=fillalpha, line=:dash,linewidth=3, color="red", label = "Baseline, to noisy observation data", xlabel="noise variance",legend = :topright,title="Full obs. and complete traj.")
plt2=plot!(noise_level_list, tmp2,ribbons=(tmp2_var, tmp2_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="blue", label = "Baseline, to ground truth data")
plt2=plot!(noise_level_list, tmp3,ribbons=(tmp3_var, tmp3_var),alpha=1,fillalpha=fillalpha,line=:dash,linewidth=3, color="orange", label = "Baseline, to expert trajectory from unseen initial state")
plt2=plot!(noise_level_list, tmp14,ribbons=(tmp14_var, tmp14_var),alpha=1,fillalpha=fillalpha, color="red",linewidth=3, label="Algorithm 1, to noisy observation data")
plt2=plot!(noise_level_list, tmp15,ribbons=(tmp15_var, tmp15_var),alpha=1,fillalpha=fillalpha, color="blue",linewidth=3, label = "Algorithm 1, to ground truth data")
plt2=plot!(noise_level_list, tmp16,ribbons=(tmp16_var, tmp16_var),alpha=1,fillalpha=fillalpha, color="orange",linewidth=3, label="Algorithm 1, to expert trajectory from unseen initial state", ylabel="distance in log scale", ylims=(-6,8))

fullplt=plot(plt1, plt2, layout=(1,2), size = (1000,400),left_margin=5Plots.mm,bottom_margin=5Plots.mm)
display(fullplt)

savefig("cars3_comparison_log.pdf")






