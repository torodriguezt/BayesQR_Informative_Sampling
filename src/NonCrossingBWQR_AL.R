bayesQR_EM <- function(y,x,w,tau_vec,mu0,sigma0,a0,b0){
  
  require(quantreg)
  source("MCMC_BWQR_AL.R")
  p <- ifelse(is.null(ncol(x)),1,ncol(x))
  q <- length(tau_vec)
  n <- length(y)
  resultado <- list()
  
  # Create auxiliary objects
  beta  <- NULL
  sigma <- NULL
  
  # Auxiliary constants
  delta2  <- 2/(tau_vec*(1-tau_vec))
  theta   <- (1-2*tau_vec)/(tau_vec*(1-tau_vec))
  
  # Set the initial values
  reg_qr <- rq(y ~ x[,-1], tau=tau_vec)
  beta   <- rbind(beta,as.vector(reg_qr$coefficients))
  sigma  <- rbind(sigma,rep(1,q))
  
  # Stop criteria
  aux_criteria <- rep(1,q)
  criteria <- 10^(-4)
  i = 2
  
  m_aux = diag(q)
  m_aux[lower.tri(m_aux)] = 1 
  X2 = m_aux %x% cbind(x,-x)
  
  initial.beta.aux <- bayesQR_weighted(y, x, w, tau_vec[1],
                                       15000, 5000, 5)
  initial.beta     <- as.vector(initial.beta.aux[[1]])
  
  while(sum(aux_criteria>criteria)>0){
    
    cov_inv   <- NULL
    y_star    <- NULL
    sigma_aux <- NULL
    
    for(k in 1:q){
      # Expectation
      a_star  <- as.vector((w*(y-x%*%beta[i-1,(p*(k-1)+1):(p*(k-1)+p)])^2)/(delta2[k]*sigma[i-1,k]))
      b_star  <- w*(2/sigma[i-1,k] + theta[k]^2/(delta2[k]*sigma[i-1,k]))
      a_star[which(a_star==0)] <- 10^(-10)
      e_nu_inverse <- sqrt(b_star)/sqrt(a_star) 
      aux_obj      <- sqrt(a_star*b_star)
      e_nu         <- NULL
      for(j in 1:n){
        e_nu[j] <- 1/e_nu_inverse[j]*(besselK(aux_obj[j], 1.5)/besselK(aux_obj[j], 0.5))  
      }
      
      # Matrices for the optimization 
      cov_inv     <- c(cov_inv,(1/(delta2[k]*sigma[i-1,k]))*(w*e_nu_inverse)) 
      y_aux       <- y-theta[k]/e_nu_inverse
      y_star      <- c(y_star,y_aux)
      
      # Maximization (scale)
      numerator   <- sum( (w*e_nu_inverse*(y_aux-x%*%beta[i-1,(p*(k-1)+1):(p*(k-1)+p)])^2)/(2*delta2[k]) ) +
        sum( (w*theta[k]^2*(e_nu-1/e_nu_inverse))/(2*delta2[k]) ) +
        sum( w*e_nu ) +
        b0
      denominator <- (3*n+a0+1)/2
      sigma_aux   <- c(sigma_aux,numerator/denominator)
      
    }
    sigma       <- rbind(sigma,sigma_aux)
    
    # Maximization (coefficients)
    sigma0.inv  <- chol2inv(chol(sigma0))
    D <- t(X2)%*%diag(cov_inv)%*%X2+diag(1,2*q)%x%sigma0.inv
    D[lower.tri(D)] <- t(D)[lower.tri(D)]
    sc = norm(D,'2')
    D = D/sc
    cholStatus <- try(u <- chol(D), silent = TRUE)
    if(class(cholStatus)[1] == "try-error"){
      D = D + diag(10^(-16),dim(D)[1])
    }
    D_chol = chol(D)
    D_chol_inv = solve(D_chol)
    
    d<- t(X2)%*%diag(cov_inv)%*%y_star+(diag(1,2*q)%x%sigma0.inv)%*%rep(mu0,2*q)
    d = d/sc
    A <- cbind(diag(1,(2*p*q)),
               rbind(matrix(0,2*p,q-1),diag(1,q-1)%x%c(1,rep(0,p-1),rep(-1,p))
               )               
    )
    b <- c(initial.beta,rep(0,dim(A)[2]-p))
    gama_aux <- solve.QP(D_chol_inv,
                         d,
                         Amat=A,
                         bvec=b,
                         meq=2*p, # meq=0 as there are no equalities
                         factorized=TRUE) 
    
    m_aux2 = diag(q)
    m_aux2[upper.tri(m_aux2)] = 1 
    aa = m_aux2 %x% rbind(diag(1,p),diag(-1,p))
    beta <- rbind(beta,c(gama_aux$solution%*%aa))
    
    aux_criteria<-c(abs(beta[i,]-beta[i-1,])^2,abs(sigma[i,]-sigma[i-1,])^2)
    i = i+1 
  }
  resultado[[1]]  <- beta[i-1,]
  resultado[[2]]  <- sigma[i-1,]
  return(resultado)
}