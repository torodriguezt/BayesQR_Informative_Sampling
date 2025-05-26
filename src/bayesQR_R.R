# Auxiliary functions for MCMC updates
# Full conditional for beta
atualizarBETA <- function(b, B, x, w, sigma, delta2, theta, v, dados) {
  B.inv  <- chol2inv(chol(B))
  covar  <- chol2inv(chol(B.inv + (1/(delta2*sigma) * (t((w/v)*x) %*% x))))
  media  <- covar %*% ((B.inv %*% b) + (1/(delta2*sigma)) * (t((w/v)*x) %*% (dados-theta*v)))
  beta   <- rmvnorm(1, media, covar)
  return(beta)
}

# Full conditional for sigma
atualizarSIGMA <- function(c, C, x, w, beta, tau2, theta, v, dados, n) {
  alpha1 <- c + 1.5*n
  beta1  <- C + sum(w*v) + (t(w*(dados-x%*%beta-theta*v)/v) %*% (dados-x%*%beta-theta*v))/(2*tau2)
  sigma  <- 1/rgamma(1, alpha1, beta1)
  return(sigma)
}

# Full conditional for the latent variable
atualizarV <- function(dados, x, w, beta, delta2, theta, sigma, N) {
  p1 <- 0.5
  p2 <- w*(dados-x%*%beta)^2/(delta2*sigma)
  p3 <- w*(2/sigma + theta^2/(delta2*sigma))
  v  <- NULL
  for(i in 1:N) {
    v[i] <- rgig(1, chi=p2[i], psi=p3[i], lambda=p1)
  }
  return(v)
}

# Bayesian Quantile Regression - MCMC
bayesQR_weighted <- function(y, x, w, tau, n_mcmc, burnin_mcmc, thin_mcmc) {
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
  v          <- rgamma(n, 2, 1)
  
  # Auxiliary constants
  delta2  <- 2/(tau*(1-tau))
  theta   <- (1-2*tau)/(tau*(1-tau))
  
  # MCMC
  for(k in 2:n_mcmc) {
    beta[k,]   <- atualizarBETA(rep(0, numcov), diag(rep(1000, numcov)), x, w, sigma[k-1,1], delta2, theta, v, y)
    v          <- atualizarV(y, x, w, beta[k,], delta2, theta, 1, n)
    sigma[k,1] <- atualizarSIGMA(0.001, 0.001, x, w, beta[k,], delta2, theta, v, y, n)
  }
  
  resultado[[1]] <- beta[seq(burnin_mcmc+1, n_mcmc, thin_mcmc),]
  resultado[[2]] <- sigma[seq(burnin_mcmc+1, n_mcmc, thin_mcmc),]
  return(resultado)
}

# Main Bayesian Quantile Regression - EM algorithm
bayesQR_EM <- function(y, x, w, tau_vec, mu0, sigma0, a0, b0) {
  require(quantreg)
  require(mvtnorm)   # For rmvnorm
  require(GIGrvg)    # For rgig
  require(quadprog)  # For solve.QP
  require(MASS)      # For norm function
  
  p <- ifelse(is.null(ncol(x)), 1, ncol(x))
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
  beta   <- rbind(beta, as.vector(reg_qr$coefficients))
  sigma  <- rbind(sigma, rep(1, q))
  
  # Stop criteria
  aux_criteria <- rep(1, q)
  criteria <- 10^(-4)
  i = 2
  
  m_aux = diag(q)
  m_aux[lower.tri(m_aux)] = 1 
  X2 = m_aux %x% cbind(x, -x)
  
  # Ejecutar MCMC con parámetros reducidos para un ejemplo rápido
  cat("Ejecutando MCMC para obtener valores iniciales...\n")
  initial.beta.aux <- bayesQR_weighted(y, x, w, tau_vec[1], 1000, 500, 2)
  
  # Extract and prepare initial beta
  if(is.matrix(initial.beta.aux[[1]])) {
    initial.beta <- colMeans(initial.beta.aux[[1]])
  } else {
    initial.beta <- initial.beta.aux[[1]]
  }
  
  cat("Dimensión de initial.beta:", length(initial.beta), "\n")
  
  while(sum(aux_criteria > criteria) > 0) {
    cov_inv   <- NULL
    y_star    <- NULL
    sigma_aux <- NULL
    
    for(k in 1:q) {
      # Expectation
      a_star  <- as.vector((w*(y-x%*%beta[i-1,(p*(k-1)+1):(p*(k-1)+p)])^2)/(delta2[k]*sigma[i-1,k]))
      b_star  <- w*(2/sigma[i-1,k] + theta[k]^2/(delta2[k]*sigma[i-1,k]))
      a_star[which(a_star==0)] <- 10^(-10)
      e_nu_inverse <- sqrt(b_star)/sqrt(a_star) 
      aux_obj      <- sqrt(a_star*b_star)
      e_nu         <- NULL
      
      for(j in 1:n) {
        e_nu[j] <- 1/e_nu_inverse[j]*(besselK(aux_obj[j], 1.5)/besselK(aux_obj[j], 0.5))  
      }
      
      # Matrices for the optimization 
      cov_inv     <- c(cov_inv, (1/(delta2[k]*sigma[i-1,k]))*(w*e_nu_inverse)) 
      y_aux       <- y-theta[k]/e_nu_inverse
      y_star      <- c(y_star, y_aux)
      
      # Maximization (scale)
      numerator   <- sum((w*e_nu_inverse*(y_aux-x%*%beta[i-1,(p*(k-1)+1):(p*(k-1)+p)])^2)/(2*delta2[k])) +
        sum((w*theta[k]^2*(e_nu-1/e_nu_inverse))/(2*delta2[k])) +
        sum(w*e_nu) +
        b0
      denominator <- (3*n+a0+1)/2
      sigma_aux   <- c(sigma_aux, numerator/denominator)
    }
    
    sigma <- rbind(sigma, sigma_aux)
    
    # Maximization (coefficients)
    sigma0.inv  <- chol2inv(chol(sigma0))
    D <- t(X2) %*% diag(cov_inv) %*% X2 + diag(1, 2*q) %x% sigma0.inv
    D[lower.tri(D)] <- t(D)[lower.tri(D)]
    sc = norm(D, '2')
    D = D/sc
    
    cholStatus <- try(u <- chol(D), silent = TRUE)
    if(class(cholStatus)[1] == "try-error") {
      D = D + diag(10^(-16), dim(D)[1])
    }
    D_chol = chol(D)
    D_chol_inv = solve(D_chol)
    
    d <- t(X2) %*% diag(cov_inv) %*% y_star + (diag(1, 2*q) %x% sigma0.inv) %*% rep(mu0, 2*q)
    d = d/sc
    
    # Fixed matrix dimension issue
    A <- cbind(diag(1, (2*p*q)),
               rbind(matrix(0, 2*p, q-1), diag(1, q-1) %x% c(1, rep(0, p-1), rep(-1, p)))
    )
    
    # Diagnóstico para depuración
    cat("Dimensions of A:", dim(A), "\n")
    
    # Utilizar solo los coeficientes para el primer cuantil (p coeficientes)
    b_vector <- rep(0, ncol(A))
    b_vector[1:p] <- initial.beta[1:p]
    
    cat("Length of b_vector:", length(b_vector), "\n")
    
    gama_aux <- solve.QP(D_chol_inv,
                         d,
                         Amat=A,
                         bvec=b_vector,
                         meq=2*p, # meq=0 as there are no equalities
                         factorized=TRUE) 
    
    m_aux2 = diag(q)
    m_aux2[upper.tri(m_aux2)] = 1 
    aa = m_aux2 %x% rbind(diag(1, p), diag(-1, p))
    beta <- rbind(beta, c(gama_aux$solution %*% aa))
    
    aux_criteria <- c(abs(beta[i,] - beta[i-1,])^2, abs(sigma[i,] - sigma[i-1,])^2)
    i = i + 1 
    
    # Add a safety check to prevent infinite loops
    if(i > 100) {
      cat("Reached maximum iterations (100). Stopping.\n")
      break
    }
  }
  
  resultado[[1]] <- beta[i-1,]
  resultado[[2]] <- sigma[i-1,]
  return(resultado)
}

# ----------------------- EJEMPLO DE USO -----------------------
# Load necessary libraries
library(mvtnorm)   # Para rmvnorm
library(GIGrvg)    # Para rgig
library(quadprog)  # Para solve.QP
library(quantreg)  # Para rq
library(MASS)      # Para mvrnorm y fractions

# Función para configurar el seed y mantener reproducibilidad
set.seed(123)

# Generar datos sintéticos más simples para probar las funciones
n <- 100  # número de observaciones
p <- 3    # número de covariables (incluyendo intercepto)

# Crear matriz de covariables X con intercepto
X <- cbind(rep(1, n), matrix(rnorm(n*(p-1)), ncol=(p-1)))
colnames(X) <- c("Intercepto", "X1", "X2")

# Definir un solo conjunto de coeficientes para simplificar
beta_true <- c(2.0, 1.5, -0.5)

# Generar errores con una distribución conocida
set.seed(456)
error <- rnorm(n, mean=0, sd=1)

# Generar variable respuesta
y <- X %*% beta_true + error

# Vector de pesos (todos iguales para este ejemplo)
w <- rep(1, n)

# Definir un solo cuantil para simplificar
tau_vec <- c(0.5)

# Parámetros previos para el modelo bayesiano
mu0 <- rep(0, p)               # Media previa para beta
sigma0 <- diag(10, p)          # Matriz de covarianza previa para beta
a0 <- 0.1                      # Parámetro de forma para sigma
b0 <- 0.1                      # Parámetro de escala para sigma

# Ejecutar el modelo tradicional primero para comparar
cat("Ejecutando modelo quantreg tradicional...\n")
qr_model <- rq(y ~ X[,-1], tau=tau_vec)
cat("Coeficientes del modelo tradicional:\n")
print(qr_model$coefficients)

# Ejecutar función bayesQR_EM con parámetros ajustados
cat("\nEjecutando Bayesian Quantile Regression...\n")
resultado <- bayesQR_EM(y, X, w, tau_vec, mu0, sigma0, a0, b0)

# Mostrar resultados
cat("\n--- Coeficientes estimados ---\n")
beta_est <- matrix(resultado[[1]], nrow=length(tau_vec), byrow=TRUE)
rownames(beta_est) <- paste0("tau=", tau_vec)
colnames(beta_est) <- c("Intercepto", "X1", "X2")
print(beta_est)

cat("\n--- Coeficientes reales ---\n")
beta_real <- matrix(beta_true, nrow=1)
rownames(beta_real) <- paste0("tau=", tau_vec)
colnames(beta_real) <- c("Intercepto", "X1", "X2")
print(beta_real)

cat("\n--- Parámetros de escala estimados ---\n")
sigmas <- resultado[[2]]
names(sigmas) <- paste0("tau=", tau_vec)
print(sigmas)