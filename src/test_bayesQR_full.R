#––– 1) Carga y define la versión original en R
library(quantreg); library(quadprog); library(GIGrvg); library(mvtnorm)

# Copia aquí el contenido de MCMC_BWQR_AL.R:
atualizarBETA <- function(b,B,x,w,sigma,delta2,theta,v,dados){
  B.inv  <- chol2inv(chol(B))
  covar  <- chol2inv(chol(B.inv + (1/(delta2*sigma))*(t((w/v)*x)%*%x)))
  media  <- covar %*% ((B.inv%*%b) + (1/(delta2*sigma))*(t((w/v)*x)%*%(dados - theta*v)))
  beta   <- rmvnorm(1,media,covar)
  return(as.vector(beta))
}
atualizarSIGMA <- function(c0,C0,x,w,beta,tau2,theta,v,dados,n){
  alpha1 <- c0 + 1.5*n
  resid  <- dados - x%*%beta - theta*v
  beta1  <- C0 + sum(w*v) + (t(w*resid/v)%*%resid)/(2*tau2)
  return(as.numeric(1/rgamma(1,alpha1,beta1)))
}
atualizarV <- function(dados,x,w,beta,delta2,theta,sigma,N){
  lambda <- 0.5
  p2     <- w*(dados - x%*%beta)^2/(delta2*sigma)
  p3     <- w*(2/sigma + theta^2/(delta2*sigma))
  v      <- numeric(N)
  for(i in seq_len(N)) v[i] <- rgig(1,lambda=lambda,chi=p2[i],psi=p3[i])
  return(v)
}
bayesQR_weighted_R <- function(y,x,w,tau,n_mcmc,burnin,thin){
  out <- list()
  fit <- rq(y~x[,-1],tau=tau)
  out[[1]] <- coef(fit)
  out[[2]] <- 1
  return(out)
}
bayesQR_EM_R <- function(y,x,w,tau_vec,mu0,sigma0,a0,b0){
  p <- ncol(x); q <- length(tau_vec); n <- length(y)
  delta2 <- 2/(tau_vec*(1-tau_vec))
  theta  <- (1-2*tau_vec)/(tau_vec*(1-tau_vec))
  reg_qr <- rq(y~x[,-1],tau=tau_vec)
  beta   <- matrix(as.vector(reg_qr$coefficients),1)
  sigma  <- matrix(1,1,q)
  crit   <- 1e-4
  aux    <- rep(1,q); i<-2
  m_aux  <- diag(q); m_aux[lower.tri(m_aux)]<-1
  X2     <- m_aux %x% cbind(x,-x)
  init   <- bayesQR_weighted_R(y,x,w,tau_vec[1],15000,5000,5)
  initb  <- init[[1]]
  while(sum(aux>crit)>0){
    cov_inv<-NULL; ystar<-NULL; sigaux<-NULL
    for(k in 1:q){
      idx <- ((p*(k-1)+1):(p*k))
      a_star <- (w*(y-x%*%beta[i-1,idx])^2)/(delta2[k]*sigma[i-1,k])
      b_star <- w*(2/sigma[i-1,k]+theta[k]^2/(delta2[k]*sigma[i-1,k]))
      a_star[a_star==0]<-1e-10
      einv<-sqrt(b_star)/sqrt(a_star)
      auxo<-sqrt(a_star*b_star)
      enu<-sapply(1:n,function(j)
        1/einv[j]*(besselK(auxo[j],1.5)/besselK(auxo[j],0.5)))
      cov_inv<-c(cov_inv,(1/(delta2[k]*sigma[i-1,k]))*(w*einv))
      y_aux<-y-theta[k]/einv; ystar<-c(ystar,y_aux)
      num<- sum((w*einv*(y_aux-x%*%beta[i-1,idx])^2)/(2*delta2[k])) +
        sum((w*theta[k]^2*(enu-1/einv))/(2*delta2[k])) +
        sum(w*enu)+b0
      den<-(3*n+a0+1)/2
      sigaux<-c(sigaux,num/den)
    }
    sigma<-rbind(sigma,sigaux)
    S0inv<-chol2inv(chol(sigma0))
    D<-t(X2)%*%diag(cov_inv)%*%X2 + diag(1,2*q)%x%S0inv
    D<-(D+t(D))/2; sc<-norm(D,'2'); D<-D/sc
    if(inherits(try(ch<-chol(D),silent=TRUE),"try-error"))
      D<-D+diag(1e-8,dim(D)[1])
    Dchol<-chol(D); Dinv<-solve(Dchol)
    d<-(t(X2)%*%diag(cov_inv)%*%ystar + (diag(1,2*q)%x%S0inv)%*%rep(mu0,2*q))/sc
    A<-cbind(diag(1,2*p*q),
             rbind(matrix(0,2*p,q-1),
                   diag(1,q-1)%x%c(1,rep(0,p-1),rep(-1,p))))
    bvec<-c(initb,rep(0,ncol(A)-length(initb)))
    sol<-solve.QP(Dinv,d,Amat=A,bvec=bvec,meq=2*p,factorized=TRUE)
    m2<-diag(q); m2[upper.tri(m2)]<-1
    aa<-m2%x%rbind(diag(1,p),diag(-1,p))
    beta<-rbind(beta,as.vector(sol$solution%*%aa))
    aux<-c((beta[i,]-beta[i-1,])^2,(sigma[i,]-sigma[i-1,])^2)
    i<-i+1
  }
  list(beta=beta[i-1,],sigma=sigma[i-1,])
}

#––– 2) Compila el C++ y prueba

library(Rcpp)
sourceCpp("bayesQR_all.cpp")

set.seed(2025)
n<-100; p<-3
X<-cbind(1,matrix(rnorm(n*(p-1)),n,p-1))
beta_true<-c(2,-1,0.5); sigma_true<-1.2
y<-X%*%beta_true + rnorm(n,0,sigma_true)
w<-rep(1,n)
tau_vec<-c(0.25,0.5,0.75)
mu0<-rep(0,p); sigma0<-diag(100,p)
a0<-0.001; b0<-0.001

# Versión R
outR <- bayesQR_EM_R(y,X,w,tau_vec,mu0,sigma0,a0,b0)
# Versión C++
outC <- bayesQR_EM(y,X,w,tau_vec,mu0,sigma0,a0,b0)

# Comparación
cat("Coef β iguales?:", all.equal(outR$beta, outC$beta, tol=1e-6), "\n")
cat("Sigma iguales?:", all.equal(outR$sigma, outC$sigma, tol=1e-6), "\n")
