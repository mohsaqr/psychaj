#!/usr/bin/env Rscript
# Generate IsingFit ground truth for numerical equivalence testing
# Uses IsingFit::IsingFit() as reference implementation

library(IsingFit)
library(jsonlite)

set.seed(42)

# ── Dataset 1: Chain structure (p=5, n=300) ──
# Generate from an Ising model with chain structure
p1 <- 5
n1 <- 300

# True interaction matrix (chain: 1-2-3-4-5)
true_weights <- matrix(0, p1, p1)
for (i in 1:(p1-1)) {
  true_weights[i, i+1] <- 0.8
  true_weights[i+1, i] <- 0.8
}

# Generate binary data using Gibbs sampling
generate_ising_data <- function(n, weights, thresholds, burnin = 1000) {
  p <- nrow(weights)
  data <- matrix(0, n, p)
  state <- rbinom(p, 1, 0.5)

  for (iter in 1:(burnin + n)) {
    for (j in 1:p) {
      eta <- thresholds[j] + sum(weights[j, -j] * state[-j])
      prob <- 1 / (1 + exp(-eta))
      state[j] <- rbinom(1, 1, prob)
    }
    if (iter > burnin) {
      data[iter - burnin, ] <- state
    }
  }
  return(data)
}

data1 <- generate_ising_data(n1, true_weights, rep(0, p1))
colnames(data1) <- paste0("V", 1:p1)

# Fit with IsingFit (AND rule, gamma=0.25)
fit1_and <- IsingFit(data1, AND = TRUE, gamma = 0.25, progressbar = FALSE, plot = FALSE)

# Fit with IsingFit (OR rule, gamma=0.25)
fit1_or <- IsingFit(data1, AND = FALSE, gamma = 0.25, progressbar = FALSE, plot = FALSE)

# ── Dataset 2: Hub structure (p=6, n=400) ──
p2 <- 6
n2 <- 400

true_weights2 <- matrix(0, p2, p2)
# Node 1 is hub connected to all others
for (i in 2:p2) {
  true_weights2[1, i] <- 0.6
  true_weights2[i, 1] <- 0.6
}

data2 <- generate_ising_data(n2, true_weights2, rep(-0.2, p2))
colnames(data2) <- paste0("V", 1:p2)

fit2_and <- IsingFit(data2, AND = TRUE, gamma = 0.25, progressbar = FALSE, plot = FALSE)

# ── Dataset 3: Independent nodes (p=4, n=200) ──
p3 <- 4
n3 <- 200
set.seed(123)
data3 <- matrix(rbinom(n3 * p3, 1, 0.5), n3, p3)
colnames(data3) <- paste0("V", 1:p3)

fit3_and <- IsingFit(data3, AND = TRUE, gamma = 0.25, progressbar = FALSE, plot = FALSE)

# ── Package results ──
results <- list(
  dataset1 = list(
    data = as.list(as.data.frame(data1)),
    n = n1,
    p = p1,
    labels = colnames(data1),
    and_rule = list(
      weightMatrix = as.matrix(fit1_and$weiadj),
      thresholds = as.numeric(fit1_and$thresholds),
      lambdas = as.numeric(fit1_and$lambda.values)
    ),
    or_rule = list(
      weightMatrix = as.matrix(fit1_or$weiadj),
      thresholds = as.numeric(fit1_or$thresholds),
      lambdas = as.numeric(fit1_or$lambda.values)
    )
  ),
  dataset2 = list(
    data = as.list(as.data.frame(data2)),
    n = n2,
    p = p2,
    labels = colnames(data2),
    and_rule = list(
      weightMatrix = as.matrix(fit2_and$weiadj),
      thresholds = as.numeric(fit2_and$thresholds),
      lambdas = as.numeric(fit2_and$lambda.values)
    )
  ),
  dataset3 = list(
    data = as.list(as.data.frame(data3)),
    n = n3,
    p = p3,
    labels = colnames(data3),
    and_rule = list(
      weightMatrix = as.matrix(fit3_and$weiadj),
      thresholds = as.numeric(fit3_and$thresholds),
      lambdas = as.numeric(fit3_and$lambda.values)
    )
  )
)

# Write JSON
outpath <- "tests/fixtures/ising-ground-truth.json"
write(toJSON(results, digits = 10, auto_unbox = TRUE), outpath)
cat("Wrote IsingFit ground truth to", outpath, "\n")

# Print summary
cat("\nDataset 1 (chain, AND):\n")
print(round(fit1_and$weiadj, 4))
cat("\nDataset 1 (chain, OR):\n")
print(round(fit1_or$weiadj, 4))
cat("\nDataset 2 (hub, AND):\n")
print(round(fit2_and$weiadj, 4))
cat("\nDataset 3 (independent, AND):\n")
print(round(fit3_and$weiadj, 4))
