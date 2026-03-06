rm(list=ls())
gc()

# Set the working directory to the Code_ACC project directory
# setwd("Code_ACC")
source("funcs_acc.R")

df_train <- simulate_data(n_sample = 2000,seed = 1, beta_star_1d = 1,case="CoxPH1d")

df_train$event %>% mean # 0.544

# set s, the batch size, be 32 64 128 256 1012
# set b, the number of samples to calculate the expectation

f_beta <- function(beta, s, b = 10000, beta_star_1d = 1, x_dist = "unifrom",divide_by_size=T){
  loss = rep(0,b)
  gradient = rep(0,b)
  hessian = rep(0,b)
  for(i in seq(b)){
    df_tmp <- simulate_data(n_sample = s,seed = NULL, case = "CoxPH1d",
                            beta_star_1d = beta_star_1d, 
                            x_dist = x_dist)
    fit_tmp <- loss_f_1d(df_tmp$x1,beta,df_tmp$event,df_tmp$time,divide_by_size=divide_by_size)
    loss[i] = fit_tmp$loss
    gradient[i] = fit_tmp$gradient
    hessian[i] = fit_tmp$hessian
    # y <- (df_tmp$time>=u)
    # x <- (df_tmp$x1)
    # value[i] <- sum(y*exp(x*beta)*x)*mean(y*exp(x*beta_star))/sum(y*exp(x*beta))
    # value[i] <- mean((-x*beta+log(sum(y*exp(x*beta))))*(y*exp(x*beta_star)))
  }
  return(list(loss = mean(loss), 
              gradient = mean(gradient),
              hessian = mean(hessian)))
}


beta_star_1d = 1
x_dist = "uniform"
b=20000
divide_by_size = T
grids_beta_batch <- expand.grid(beta = seq(from = (beta_star_1d-0.1), to= (beta_star_1d+0.1), by = 0.02),
                                batch_size = c(32,64,128,256,512)) %>% 
  mutate(loss = 0,
         gradient = 0,
         hessian = 0)

pb <- txtProgressBar(min = 0, max = nrow(grids_beta_batch), style = 3)
for(i in seq(nrow(grids_beta_batch))){
  result_tmp = f_beta(grids_beta_batch$beta[i],
                      grids_beta_batch$batch_size[i],
                      b=b, 
                      beta_star_1d = beta_star_1d, 
                      x_dist = x_dist,
                      divide_by_size=divide_by_size)
  grids_beta_batch$loss[i] <- result_tmp$loss
  grids_beta_batch$gradient[i] <- result_tmp$gradient
  grids_beta_batch$hessian[i] <- result_tmp$hessian
  setTxtProgressBar(pb, i)
}

grids_beta_batch$`batch size` <- as.factor(grids_beta_batch$batch_size)
grids_beta_batch <- grids_beta_batch %>% 
  group_by(`batch size`) %>% 
  mutate(loss_min = min(loss)) %>% 
  mutate(loss_adj = loss-loss_min) %>% 
  ungroup

p_gradient <- ggplot(grids_beta_batch, aes(x = beta, y = gradient, group = `batch size`,col = `batch size`))+
  geom_line()+
  geom_point()+
  labs(
    x = expression(theta),
    y = expression("E["~symbol("\xd1")[theta]~L[Cox]^(s)~(theta)~"]"),
    color='Batch Size'
  )+
  # scale_colour_brewer(palette = "RdYlBu")+
  scale_color_manual(values=c("#d94c66","#EFAAC7","#f2dc70","#61B8C9","#7070D5","#D5D5E7"))+
  theme_bw()+ scale_fill_discrete(name = "New Legend Title")+
  theme(legend.position = "inside", legend.position.inside = c(0.9, 0.32),
        legend.title = element_text(size = 8), 
        legend.text  = element_text(size = 8))
p_gradient



ggsave(filename = "Figure/simulation_population_gradient.png", p_gradient,
       width = 8, height = 3, dpi = 300, device='png')