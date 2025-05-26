# Full conditional for beta
atualizarBETA<-function(b,B,x,w,sigma,delta2,theta,v,dados){
  B.inv  <- chol2inv(chol(B))
  covar  <- chol2inv(chol(B.inv+(1/(delta2*sigma)*(t((w/v)*x)%*%x))))
  media  <- covar%*%((B.inv%*%b)+(1/(delta2*sigma))*(t((w/v)*x)%*%(dados-theta*v)))
  beta   <- rmvnorm(1,media,covar)
  return(beta)
}

# Full conditional for sigma
atualizarSIGMA<-function(c,C,x,w,beta,tau2,theta,v,dados,n){
  alpha1 <- c + 1.5*n
  beta1  <- C + sum(w*v) + (t(w*(dados-x%*%beta-theta*v)/v)%*%(dados-x%*%beta-theta*v))/(2*tau2)
  sigma  <- 1/rgamma(1, alpha1, beta1)
  return(sigma)
}

# Full conditional for the latent variable
atualizarV<-function(dados,x,w,beta,delta2,theta,sigma,N){
  p1 <- 0.5
  p2 <- w*(dados-x%*%beta)^2/(delta2*sigma)
  p3 <- w*(2/sigma + theta^2/(delta2*sigma))
  v  <- NULL
  for(i in 1:N){
    v[i]  <- rgig(1, chi=p2[i], psi=p3[i], lambda=p1)
  }
  return(v)
}

# Bayesian Quantile Regression - MCMC
bayesQR_weighted <- function(y,x,w,tau,n_mcmc,burnin_mcmc,thin_mcmc){
  n         <- length(y)
  numcov    <- ncol(x)
  resultado <- list()
  
  # Create auxiliary objects
  beta  <- matrix(NA, n_mcmc, numcov)
  sigma <- matrix(NA, n_mcmc, 1)
  # Set the initial values
  reg_lm     <- lm(y~-1+x)
  beta[1,]   <- reg_lm$coefficients
  sigma[1,1] <- 1
  v          <- rgamma(n,2,1)
  # Auxiliary constants
  delta2  <- 2/(tau*(1-tau))
  theta   <- (1-2*tau)/(tau*(1-tau))
  # MCMC
  for(k in 2:n_mcmc){
    beta[k,]   <- atualizarBETA(rep(0,numcov),diag(rep(1000,numcov)),x,w,sigma[k-1,1],delta2,theta,v,y)
    v          <- atualizarV(y,x,w,beta[k,],delta2,theta,1,n)
    sigma[k,1]   <- atualizarSIGMA(0.001,0.001,x,w,beta[k,],delta2,theta,v,y,n)
  }
  resultado[[1]]  <- beta[seq(burnin_mcmc+1,n_mcmc,thin_mcmc),]
  resultado[[2]]  <- sigma[seq(burnin_mcmc+1,n_mcmc,thin_mcmc),]
  
  return(resultado)
}