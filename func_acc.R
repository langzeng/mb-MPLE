require(ggplot2)
require(dplyr)
require(gtsummary)
require(survival)
require(tidyr)
require(glue)
require(survminer)
select <- dplyr::select
filter <- dplyr::filter


simulate_data <- function(n_sample,
                          # exponential baseline hazard distribution
                          mu = 45,
                          # pi_centered = T,
                          seed = 133,
                          case = 1, 
                          beta_star_1d = NULL,
                          beta_star_2d = NULL,
                          x_dist = "uniform"){
  if(!is.null(seed)){
    set.seed(seed)
  }
  
  if(case != "CoxPH1d"){
    x1 <- runif(n_sample,0,2)
    x2 <- runif(n_sample,0,2)
    x3 <- runif(n_sample,0,2)
    x4 <- runif(n_sample,0,2)
    x5 <- runif(n_sample,0,2)
  }
  if(case == "CoxPH1d"){
    if(x_dist == "uniform"){x1 <- runif(n_sample,0,10)}
    if(x_dist == "exp"){x1 <- rexp(n_sample,1)}
    if(x_dist == "norm"){x1 <- rnorm(n_sample)}
  }
  

  if(case == 2){
    rs <- ((x1^2)*(x2^3)+log(x3+1)+sqrt(x4*x5+1)+exp(x5/2))^2/20-6
  }
  if(case == 1){
    rs <- x1+x2+x3+x4+x5-5  
  }
  if(case == "CoxPH1d"){
    rs <- beta_star_1d*x1
  }
  if(case == "CoxPH2d"){
    rs <- beta_star_2d[1]*x1+beta_star_2d[2]*x2-5
  }
  
  u_vector <- runif(n_sample)
  c_vector <- rexp(n_sample,rate=(1/mu))
  tt_vector <- sqrt(-log(1-u_vector)*20/exp(rs))
  t_vector <- pmin(tt_vector,c_vector)
  event_vector <- ifelse(tt_vector<=c_vector,1,0)
  
  if(case == "CoxPH1d"){
    df <- data.frame(id = seq(n_sample),
                     x1 = x1,
                     rs_true = rs,
                     time0 = 0,
                     time = t_vector,
                     event = event_vector)
  }else{
    df <- data.frame(id = seq(n_sample),
                     x1 = x1,
                     x2 = x2,
                     x3 = x3,
                     x4 = x4,
                     x5 = x5,
                     rs_true = rs,
                     time0 = 0,
                     time = t_vector,
                     event = event_vector)
  }
  return(df)
}


loss_f_1d <- function(x,beta,event,time,divide_by_size = T){
  if(divide_by_size){
    s <- length(event)
  }else{
    s <- sum(event)
  }
  event_time <- time[event == 1]
  at_risk_index <- outer(time,event_time,">=") %>% as.matrix
  rs <- x*beta
  at_risk_exprs <- as.vector(matrix(exp(rs),nrow = 1) %*% at_risk_index)
  at_risk_exprsx <- as.vector(matrix(exp(rs)*x,nrow = 1) %*% at_risk_index)
  at_risk_exprsx2 <- as.vector(matrix(exp(rs)*x*x,nrow = 1) %*% at_risk_index)
  loss <- -mean(rs*event)+sum(log(at_risk_exprs))/s
  gradient <- -mean(x*event)+sum(at_risk_exprsx/at_risk_exprs)/s
  hessian <- sum(at_risk_exprsx2/at_risk_exprs-(at_risk_exprsx/at_risk_exprs)^2)/s
  return(list(loss = loss, gradient = gradient, hessian=hessian))
}

loss_tdCoxSNN = function(y_true, y_pred){
  y_true = tf$cast(y_true, tf$float32) # tstart, tstop, event
  y_pred = tf$cast(y_pred, tf$float32)
  y_pred = tf$squeeze(y_pred)

  time0 = tf$cast(tf$squeeze(y_true[,1]),tf$float32)
  time = tf$cast(tf$squeeze(y_true[,2]),tf$float32)
  event = tf$cast(tf$squeeze(y_true[,3]),tf$float32)
  
  no_event = tf$cond(tf$equal(tf$reduce_sum(event),tf$cast(0,tf$float32)), function(){return(tf$cast(1,tf$float32))},function(){return(tf$cast(0,tf$float32))})
  
  event = tf$add(event,no_event)
  sort_index = tf$argsort(time)
  time0 = tf$gather(params = time0, indices = sort_index)
  time = tf$gather(params = time, indices = sort_index)
  event = tf$gather(params = event, indices = sort_index)
  y_pred = tf$gather(params = y_pred, indices = sort_index)
  
  time_event = time * event
  positive_indexes = (tf$where(tf$greater(time_event, tf$zeros_like(time_event))))
  alleventtime = tf$gather(time_event,positive_indexes)
  
  loc = tf$where(time_event>0,tf$ones_like(time_event,dtype=tf$bool),tf$zeros_like(time_event,dtype=tf$bool))
  loc$set_shape(list(NULL))
  alleventtime = tf$boolean_mask(time_event,loc)
  
  unique_w_c = tf$unique_with_counts(alleventtime)
  eventtime = unique_w_c[0]
  tie_count = unique_w_c[2]
  
  at_risk_index = tf$where(tf$logical_and(tf$less(time0,tf$expand_dims(eventtime,tf$cast(1,tf$int32))),tf$greater_equal(time,tf$expand_dims(eventtime,tf$cast(1,tf$int32)))),
                           1., 0.)
  event_index = tf$where(tf$equal(time,tf$expand_dims(eventtime,tf$cast(1,tf$int32))),1.,0.)
  
  # haz = exp(risk)
  tie_haz = tf$matmul(event_index,tf$expand_dims(tf$exp(tf$clip_by_value(y_pred,-20,20))*event,tf$cast(1,tf$int32)))
  
  tie_risk = tf$matmul(event_index,tf$expand_dims(y_pred*event,tf$cast(1,tf$int32)))
  
  cum_haz = tf$matmul(at_risk_index,
                      tf$expand_dims(tf$exp(tf$clip_by_value(y_pred,-20,20)),tf$cast(1,tf$int32)))
  
  mask_tie_haz = tf$less(tf$range(tf$reduce_max(tie_count)),
                         tf$expand_dims(tie_count-1,tf$cast(1,tf$int32)))
  mask_tie_risk = tf$less(tf$range(tf$reduce_max(tie_count)),
                          tf$expand_dims(tie_count,tf$cast(1,tf$int32)))
  
  out0 = tf$zeros_like(mask_tie_haz,dtype = tf$float32)
  out1 = tf$cast(tf$cumsum(tf$ones_like(mask_tie_haz,dtype = tf$float32), tf$cast(1,tf$int32)),dtype = tf$float32)
  out = tf$where(mask_tie_haz, out1, out0)
  tie_count_matrix = tf$expand_dims(tf$cast(tie_count, dtype = tf$float32),tf$cast(1,tf$int32))
  
  J = tf$divide(out,tie_count_matrix)
  efron_correction = J*tie_haz
  log_sum_haz0 = tf$where(mask_tie_risk,
                          tf$ones_like(mask_tie_risk,dtype = tf$float32),
                          out0)*cum_haz
  log_sum_haz = tf$where(mask_tie_risk,
                         tf$math$log(log_sum_haz0-efron_correction+tf$cast(1e-15,tf$float32)),
                         out0)
  log_sum_haz_value = tf$reduce_sum(log_sum_haz)
  log_lik = tf$reduce_sum(tie_risk)-log_sum_haz_value
  
  log_lik_output = tf$multiply(tf$negative(log_lik),tf$subtract(1,no_event))
  
  return(log_lik_output)
}

loss_tdCoxSNN_bysize = function(y_true,y_pred){
  n = tf$reduce_sum(tf$ones_like(tf$squeeze(tf$cast(y_pred, tf$float32)),dtype=tf$float32))
  out = loss_tdCoxSNN(y_true,y_pred)/n
  
  return(out)
}

loss_tdCoxSNN_byevent = function(y_true,y_pred){
  event = tf$cast(tf$squeeze(y_true[,3]),tf$float32)
  nevent = tf$reduce_sum(event)
  out = loss_tdCoxSNN(y_true,y_pred)/nevent
  
  return(out)
}

loss_f_samplek <- function(x,beta,event,time,sample_k = k){
  event_time <- time[event == 1]
  event_index <- outer(time,event_time,"==") %>% as.matrix
  live_longer_index <- outer(time,event_time,">") %>% as.matrix
  
  live_longer_index_sampled <- apply(live_longer_index,2,function(col){
    col[sample(which(col),k)]
  })
  
  rs <- x*beta
  at_risk_exprs <- as.vector(matrix(exp(rs),nrow = 1) %*% at_risk_index)
  loss <- -mean(rs*event)+sum(log(at_risk_exprs))/length(event)
  return(loss)
}

paired_loss <- function(beta,x1,time1,event1,x2,time2,event2){
  if(event1+event2 == 0){return(0)}
  else{
    if(time1 > time2){
      x_tmp = x1
      time_tmp = time1
      event_tmp = event1
      x1 = x2
      time1 = time2
      event1 = event2
      x2 = x_tmp
      time2 = time_tmp
      event2 = event_tmp
    }
    if(event1 == 0){return(0)}
    else{
      out = sum(x1*beta)-log(exp(sum(x1*beta))+exp(sum(x2*beta)))
      return(-out)
    }
  }
}

paired_gradient <- function(beta,x1,time1,event1,x2,time2,event2){
  if(event1+event2 == 0){return(0)}
  else{
    if(time1 > time2){
      x_tmp = x1
      time_tmp = time1
      event_tmp = event1
      x1 = x2
      time1 = time2
      event1 = event2
      x2 = x_tmp
      time2 = time_tmp
      event2 = event_tmp
    }
    if(event1 == 0){return(0)}
    else{
      out = x1-(exp(sum(x1*beta))*x1+exp(sum(x2*beta))*x2)/(exp(sum(x1*beta))+exp(sum(x2*beta)))
      return(-out)
    }
  }
}

x_scale <- function(x,mean = x_train_scale1, sd = x_train_scale2){
  x = unlist(x)
  return((x-mean)/sd)
}