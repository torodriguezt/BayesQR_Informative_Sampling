bayesQR_EM_R <- function(y, x, w, tau_vec, mu0, sigma0, a0, b0) {
  require(quantreg)
  require(quadprog)
  
  p <- if (is.null(ncol(x))) 1 else ncol(x)
  q <- length(tau_vec)
  n <- length(y)
  
  # Constantes auxiliares
  delta2 <- 2 / (tau_vec * (1 - tau_vec))
  theta  <- (1 - 2 * tau_vec) / (tau_vec * (1 - tau_vec))
  
  # Inicialización con regresión cuantílica multicuantil
  reg_qr <- rq(y ~ x[,-1], tau = tau_vec)
  beta   <- matrix(as.vector(reg_qr$coefficients), nrow = 1)
  sigma  <- matrix(1, nrow = 1, ncol = q)
  
  aux_criteria <- rep(1, q)
  criteria     <- 1e-4
  i <- 2
  
  # Construir X2 = kron(m_aux, [x | -x])
  m_aux <- diag(q)
  m_aux[lower.tri(m_aux)] <- 1
  X2 <- m_aux %x% cbind(x, -x)
  
  # Obtener beta inicial (solo la primera fila) desde MCMC auxiliar
  init    <- bayesQR_weighted(y, x, w, tau_vec[1], 15000, 5000, 5)
  initial.beta <- as.vector(init[[1]][1, ])  # <-- CORRECCIÓN AQUÍ
  
  while (sum(aux_criteria > criteria) > 0) {
    cov_inv   <- numeric(0)
    y_star    <- numeric(0)
    sigma_aux <- numeric(0)
    
    # E‐step y escala
    for (k in seq_len(q)) {
      idx    <- ((p*(k-1)+1):(p*k))
      resid2 <- (y - x %*% beta[i-1, idx])^2
      a_star <- (w * resid2) / (delta2[k] * sigma[i-1, k])
      b_star <- w * (2 / sigma[i-1, k] + theta[k]^2 / (delta2[k] * sigma[i-1, k]))
      a_star[a_star == 0] <- 1e-10
      e_inv  <- sqrt(b_star) / sqrt(a_star)
      auxo   <- sqrt(a_star * b_star)
      e_nu   <- sapply(seq_len(n), function(j)
        1 / e_inv[j] * (besselK(auxo[j], 1.5) / besselK(auxo[j], 0.5))
      )
      
      cov_inv   <- c(cov_inv, (1/(delta2[k] * sigma[i-1, k])) * (w * e_inv))
      y_aux     <- y - theta[k] / e_inv
      y_star    <- c(y_star, y_aux)
      
      num <- sum((w * e_inv * (y_aux - x %*% beta[i-1, idx])^2) / (2 * delta2[k])) +
        sum((w * theta[k]^2 * (e_nu - 1/e_inv)) / (2 * delta2[k])) +
        sum(w * e_nu) + b0
      den <- (3 * n + a0 + 1) / 2
      sigma_aux <- c(sigma_aux, num/den)
    }
    sigma <- rbind(sigma, sigma_aux)
    
    # M‐step coeficientes
    sigma0.inv <- chol2inv(chol(sigma0))
    D <- t(X2) %*% diag(cov_inv) %*% X2 + diag(1, 2*q) %x% sigma0.inv
    D <- (D + t(D)) / 2
    sc <- norm(D, "2")
    D  <- D / sc
    
    # Cholesky con jitter si es necesario
    if (inherits(try(ch <- chol(D), silent = TRUE), "try-error")) {
      D <- D + diag(1e-8, nrow(D))
    }
    D_chol     <- chol(D)
    D_chol_inv <- solve(D_chol)
    
    d <- t(X2) %*% diag(cov_inv) %*% y_star +
      (diag(1,2*q) %x% sigma0.inv) %*% rep(mu0, 2*q)
    d <- d / sc
    
    A <- cbind(
      diag(1, 2*p*q),
      rbind(
        matrix(0, 2*p, q-1),
        diag(1, q-1) %x% c(1, rep(0, p-1), rep(-1, p))
      )
    )
    # ← CONSTRUCCIÓN CORRECTA DE bvec
    bvec <- c(initial.beta, rep(0, ncol(A) - length(initial.beta)))
    
    sol <- solve.QP(D_chol_inv, d, Amat = A, bvec = bvec,
                    meq = 2*p, factorized = TRUE)
    
    m2 <- diag(q); m2[upper.tri(m2)] <- 1
    aa <- m2 %x% rbind(diag(1, p), diag(-1, p))
    beta <- rbind(beta, as.vector(sol$solution %*% aa))
    
    aux_criteria <- c((beta[i,] - beta[i-1,])^2,
                      (sigma[i,] - sigma[i-1,])^2)
    i <- i + 1
  }
  
  list(beta = beta[i-1, ], sigma = sigma[i-1, ])
}



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