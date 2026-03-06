#!/usr/bin/env Rscript
# Generate MGM ground truth for numerical equivalence testing
# Uses mgm::mgm() as reference implementation

library(mgm)
library(jsonlite)

set.seed(42)

# ── Dataset 1: All-gaussian (p=4, n=300) ──
n1 <- 300
p1 <- 4

# Correlated gaussian data
Sigma1 <- matrix(c(
  1.0, 0.6, 0.0, 0.0,
  0.6, 1.0, 0.4, 0.0,
  0.0, 0.4, 1.0, 0.3,
  0.0, 0.0, 0.3, 1.0
), p1, p1)
L1 <- chol(Sigma1)
data1 <- matrix(rnorm(n1 * p1), n1, p1) %*% L1
colnames(data1) <- paste0("V", 1:p1)

fit1 <- mgm(data = data1,
            type = rep("g", p1),
            level = rep(1, p1),
            lambdaSel = "EBIC",
            lambdaGam = 0.25,
            pbar = FALSE)

# ── Dataset 2: All-binary (p=4, n=400, chain) ──
n2 <- 400
p2 <- 4

# Chain binary data
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

data2 <- generate_binary_chain(n2, p2, 3, 42)
colnames(data2) <- paste0("V", 1:p2)

fit2 <- mgm(data = data2,
            type = rep("c", p2),
            level = rep(2, p2),
            lambdaSel = "EBIC",
            lambdaGam = 0.25,
            pbar = FALSE)

# ── Dataset 3: Mixed data (p=5: 2 gaussian + 2 binary + 1 poisson, n=500) ──
n3 <- 500
set.seed(42)

# Generate correlated gaussian
z1 <- rnorm(n3)
z2 <- 0.6 * z1 + 0.8 * rnorm(n3)

# Binary depends on z1
eta3 <- 0.8 * z1
x3 <- rbinom(n3, 1, 1 / (1 + exp(-eta3)))

# Independent binary
x4 <- rbinom(n3, 1, 0.5)

# Poisson depends on z2
x5 <- rpois(n3, exp(0.5 + 0.3 * z2))

data3 <- cbind(z1, z2, x3, x4, x5)
colnames(data3) <- c("gauss1", "gauss2", "binary1", "binary2", "count1")

fit3 <- mgm(data = data3,
            type = c("g", "g", "c", "c", "p"),
            level = c(1, 1, 2, 2, 1),
            lambdaSel = "EBIC",
            lambdaGam = 0.25,
            pbar = FALSE)

# ── Package results ──
# mgm stores results differently:
# fit$pairwise$wadj = weighted adjacency matrix
# fit$pairwise$signs = sign matrix

results <- list(
  dataset1_all_gaussian = list(
    data = as.list(as.data.frame(data1)),
    n = n1,
    p = p1,
    labels = colnames(data1),
    types = rep("g", p1),
    levels = rep(1, p1),
    weightMatrix = fit1$pairwise$wadj,
    signMatrix = fit1$pairwise$signs,
    lambdas = if (!is.null(fit1$nodemodels)) {
      sapply(1:p1, function(j) {
        nm <- fit1$nodemodels[[j]]
        if (!is.null(nm$lambda)) nm$lambda else NA
      })
    } else { rep(NA, p1) }
  ),
  dataset2_all_binary = list(
    data = as.list(as.data.frame(data2)),
    n = n2,
    p = p2,
    labels = colnames(data2),
    types = rep("c", p2),
    levels = rep(2, p2),
    weightMatrix = fit2$pairwise$wadj,
    signMatrix = fit2$pairwise$signs
  ),
  dataset3_mixed = list(
    data = as.list(as.data.frame(data3)),
    n = n3,
    p = 5,
    labels = colnames(data3),
    types = c("g", "g", "c", "c", "p"),
    levels = c(1, 1, 2, 2, 1),
    weightMatrix = fit3$pairwise$wadj,
    signMatrix = fit3$pairwise$signs
  )
)

# Write JSON
outpath <- "tests/fixtures/mgm-ground-truth.json"
write(toJSON(results, digits = 10, auto_unbox = TRUE), outpath)
cat("Wrote MGM ground truth to", outpath, "\n")

# Print summary
cat("\nDataset 1 (all-gaussian) weights:\n")
print(round(fit1$pairwise$wadj, 4))
cat("\nDataset 1 (all-gaussian) signs:\n")
print(fit1$pairwise$signs)

cat("\nDataset 2 (all-binary) weights:\n")
print(round(fit2$pairwise$wadj, 4))

cat("\nDataset 3 (mixed) weights:\n")
print(round(fit3$pairwise$wadj, 4))
cat("\nDataset 3 (mixed) signs:\n")
print(fit3$pairwise$signs)
