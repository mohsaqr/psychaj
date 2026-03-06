#!/usr/bin/env Rscript
# Extended MGM ground truth — more datasets, more comparisons
library(mgm)
library(jsonlite)

results <- list()

# ── Dataset 4: Gaussian with known partial correlations (p=5, n=500) ──
set.seed(101)
n <- 500; p <- 5
# Build precision matrix (tridiagonal → chain partial correlations)
Theta <- diag(p)
for (i in 1:(p-1)) { Theta[i, i+1] <- -0.3; Theta[i+1, i] <- -0.3 }
Sigma <- solve(Theta)
L <- chol(Sigma)
data4 <- matrix(rnorm(n * p), n, p) %*% L
colnames(data4) <- paste0("V", 1:p)

fit4 <- mgm(data = data4, type = rep("g", p), level = rep(1, p),
            lambdaSel = "EBIC", lambdaGam = 0.25, pbar = FALSE)
results$dataset4_gaussian_chain <- list(
  description = "Gaussian chain from precision matrix (p=5, n=500)",
  data = as.list(as.data.frame(data4)),
  n = n, p = p, labels = colnames(data4),
  types = rep("g", p), levels = rep(1, p),
  weightMatrix = fit4$pairwise$wadj,
  signMatrix = fit4$pairwise$signs
)

# ── Dataset 5: Binary chain (p=6, n=500) with IsingFit cross-check ──
set.seed(102)
n <- 500; p <- 6
generate_binary_chain <- function(n, p, strength, seed) {
  set.seed(seed)
  data <- matrix(0, n, p)
  for (i in 1:n) {
    data[i, 1] <- rbinom(1, 1, 0.5)
    for (j in 2:p) {
      eta <- strength * (data[i, j-1] - 0.5)
      prob <- 1 / (1 + exp(-eta))
      data[i, j] <- rbinom(1, 1, prob)
    }
  }
  return(data)
}
data5 <- generate_binary_chain(n, p, 2.5, 102)
colnames(data5) <- paste0("V", 1:p)

fit5 <- mgm(data = data5, type = rep("c", p), level = rep(2, p),
            lambdaSel = "EBIC", lambdaGam = 0.25, pbar = FALSE)
results$dataset5_binary_chain6 <- list(
  description = "Binary chain (p=6, n=500, strength=2.5)",
  data = as.list(as.data.frame(data5)),
  n = n, p = p, labels = colnames(data5),
  types = rep("c", p), levels = rep(2, p),
  weightMatrix = fit5$pairwise$wadj,
  signMatrix = fit5$pairwise$signs
)

# ── Dataset 6: Mixed 3 types (p=6: 2g + 2c + 2p, n=600) ──
set.seed(103)
n <- 600
z1 <- rnorm(n)
z2 <- 0.5 * z1 + sqrt(1 - 0.25) * rnorm(n)
eta3 <- 0.7 * z1 + 0.3 * z2
x3 <- rbinom(n, 1, 1 / (1 + exp(-eta3)))
x4 <- rbinom(n, 1, 1 / (1 + exp(-0.5 * z2)))
x5 <- rpois(n, exp(0.3 + 0.2 * z1))
x6 <- rpois(n, exp(0.5 + 0.25 * z2 + 0.1 * x3))
data6 <- cbind(z1, z2, x3, x4, x5, x6)
colnames(data6) <- c("g1", "g2", "b1", "b2", "p1", "p2")

fit6 <- mgm(data = data6, type = c("g","g","c","c","p","p"),
            level = c(1,1,2,2,1,1),
            lambdaSel = "EBIC", lambdaGam = 0.25, pbar = FALSE)
results$dataset6_mixed_6node <- list(
  description = "Mixed 6-node (2g+2c+2p, n=600)",
  data = as.list(as.data.frame(data6)),
  n = n, p = 6, labels = colnames(data6),
  types = c("g","g","c","c","p","p"), levels = c(1,1,2,2,1,1),
  weightMatrix = fit6$pairwise$wadj,
  signMatrix = fit6$pairwise$signs
)

# ── Dataset 7: Gaussian with varying correlation strengths (p=4, n=400) ──
set.seed(104)
n <- 400; p <- 4
# Strong, medium, weak, zero partial correlations
Sigma7 <- matrix(c(1, 0.8, 0.3, 0,
                    0.8, 1, 0.5, 0.1,
                    0.3, 0.5, 1, 0.7,
                    0, 0.1, 0.7, 1), p, p)
L7 <- chol(Sigma7)
data7 <- matrix(rnorm(n * p), n, p) %*% L7
colnames(data7) <- paste0("V", 1:p)

fit7 <- mgm(data = data7, type = rep("g", p), level = rep(1, p),
            lambdaSel = "EBIC", lambdaGam = 0.25, pbar = FALSE)
results$dataset7_gaussian_varied <- list(
  description = "Gaussian with varying correlation strengths (p=4, n=400)",
  data = as.list(as.data.frame(data7)),
  n = n, p = p, labels = colnames(data7),
  types = rep("g", p), levels = rep(1, p),
  weightMatrix = fit7$pairwise$wadj,
  signMatrix = fit7$pairwise$signs
)

# ── Dataset 8: Large n mixed (p=4, n=1000) ──
set.seed(105)
n <- 1000
z1 <- rnorm(n)
x2 <- rbinom(n, 1, 1 / (1 + exp(-0.6 * z1)))
x3 <- rpois(n, exp(0.4 + 0.3 * z1))
z4 <- 0.4 * z1 + sqrt(1 - 0.16) * rnorm(n)
data8 <- cbind(z1, x2, x3, z4)
colnames(data8) <- c("gauss1", "binary1", "count1", "gauss2")

fit8 <- mgm(data = data8, type = c("g","c","p","g"), level = c(1,2,1,1),
            lambdaSel = "EBIC", lambdaGam = 0.25, pbar = FALSE)
results$dataset8_large_n_mixed <- list(
  description = "Large-n mixed (p=4, n=1000)",
  data = as.list(as.data.frame(data8)),
  n = n, p = 4, labels = colnames(data8),
  types = c("g","c","p","g"), levels = c(1,2,1,1),
  weightMatrix = fit8$pairwise$wadj,
  signMatrix = fit8$pairwise$signs
)

# ── Dataset 9: All-poisson (p=4, n=400) ──
set.seed(106)
n <- 400
x1 <- rpois(n, 3)
x2 <- rpois(n, exp(0.5 + 0.1 * x1))
x3 <- rpois(n, 2)
x4 <- rpois(n, exp(0.3 + 0.05 * x2 + 0.08 * x3))
data9 <- cbind(x1, x2, x3, x4)
colnames(data9) <- paste0("P", 1:4)

fit9 <- mgm(data = data9, type = rep("p", 4), level = rep(1, 4),
            lambdaSel = "EBIC", lambdaGam = 0.25, pbar = FALSE)
results$dataset9_all_poisson <- list(
  description = "All-poisson (p=4, n=400)",
  data = as.list(as.data.frame(data9)),
  n = n, p = 4, labels = colnames(data9),
  types = rep("p", 4), levels = rep(1, 4),
  weightMatrix = fit9$pairwise$wadj,
  signMatrix = fit9$pairwise$signs
)

write(toJSON(results, digits = 10, auto_unbox = TRUE), "tests/fixtures/mgm-extended-ground-truth.json")
cat("Wrote extended MGM ground truth\n\n")

for (name in names(results)) {
  ds <- results[[name]]
  cat(sprintf("=== %s: %s ===\n", name, ds$description))
  cat("  Weights:\n")
  print(round(ds$weightMatrix, 4))
  if (!is.null(ds$signMatrix)) {
    cat("  Signs:\n")
    print(ds$signMatrix)
  }
  cat("\n")
}
