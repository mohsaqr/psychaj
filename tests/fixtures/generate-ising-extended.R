#!/usr/bin/env Rscript
# Extended IsingFit ground truth — more datasets, more metrics
library(IsingFit)
library(jsonlite)

generate_ising_data <- function(n, weights, thresholds, burnin = 2000, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  p <- nrow(weights)
  data <- matrix(0, n, p)
  state <- rbinom(p, 1, 0.5)
  for (iter in 1:(burnin + n)) {
    for (j in 1:p) {
      eta <- thresholds[j] + sum(weights[j, -j] * state[-j])
      prob <- 1 / (1 + exp(-eta))
      state[j] <- rbinom(1, 1, prob)
    }
    if (iter > burnin) data[iter - burnin, ] <- state
  }
  return(data)
}

results <- list()

# ── Dataset 4: Dense network (p=4, n=500, all connected) ──
set.seed(100)
p <- 4; n <- 500
W4 <- matrix(c(0, 0.7, 0.5, 0.3,
               0.7, 0, 0.6, 0.4,
               0.5, 0.6, 0, 0.8,
               0.3, 0.4, 0.8, 0), p, p)
data4 <- generate_ising_data(n, W4, rep(0, p), seed = 100)
colnames(data4) <- paste0("V", 1:p)
fit4_and <- IsingFit(data4, AND = TRUE, gamma = 0.25, progressbar = FALSE, plot = FALSE)
fit4_or <- IsingFit(data4, AND = FALSE, gamma = 0.25, progressbar = FALSE, plot = FALSE)

results$dataset4_dense <- list(
  description = "Dense fully connected (p=4, n=500)",
  data = as.list(as.data.frame(data4)),
  n = n, p = p, labels = colnames(data4),
  and_rule = list(weightMatrix = as.matrix(fit4_and$weiadj), thresholds = as.numeric(fit4_and$thresholds)),
  or_rule = list(weightMatrix = as.matrix(fit4_or$weiadj), thresholds = as.numeric(fit4_or$thresholds))
)

# ── Dataset 5: Larger chain (p=8, n=500) ──
set.seed(200)
p <- 8; n <- 500
W5 <- matrix(0, p, p)
for (i in 1:(p-1)) { W5[i, i+1] <- 0.9; W5[i+1, i] <- 0.9 }
data5 <- generate_ising_data(n, W5, rep(0, p), seed = 200)
colnames(data5) <- paste0("V", 1:p)
fit5 <- IsingFit(data5, AND = TRUE, gamma = 0.25, progressbar = FALSE, plot = FALSE)

results$dataset5_large_chain <- list(
  description = "Large chain (p=8, n=500)",
  data = as.list(as.data.frame(data5)),
  n = n, p = p, labels = colnames(data5),
  and_rule = list(weightMatrix = as.matrix(fit5$weiadj), thresholds = as.numeric(fit5$thresholds))
)

# ── Dataset 6: Two clusters (p=6, n=400) ──
set.seed(300)
p <- 6; n <- 400
W6 <- matrix(0, p, p)
# Cluster 1: nodes 1-3
W6[1,2] <- W6[2,1] <- 0.8
W6[1,3] <- W6[3,1] <- 0.6
W6[2,3] <- W6[3,2] <- 0.7
# Cluster 2: nodes 4-6
W6[4,5] <- W6[5,4] <- 0.9
W6[4,6] <- W6[6,4] <- 0.5
W6[5,6] <- W6[6,5] <- 0.7
# Weak bridge
W6[3,4] <- W6[4,3] <- 0.3
data6 <- generate_ising_data(n, W6, rep(0, p), seed = 300)
colnames(data6) <- paste0("V", 1:p)
fit6 <- IsingFit(data6, AND = TRUE, gamma = 0.25, progressbar = FALSE, plot = FALSE)

results$dataset6_two_clusters <- list(
  description = "Two clusters with bridge (p=6, n=400)",
  data = as.list(as.data.frame(data6)),
  n = n, p = p, labels = colnames(data6),
  and_rule = list(weightMatrix = as.matrix(fit6$weiadj), thresholds = as.numeric(fit6$thresholds))
)

# ── Dataset 7: Unbalanced prevalence (p=5, n=300, nonzero thresholds) ──
set.seed(400)
p <- 5; n <- 300
W7 <- matrix(0, p, p)
W7[1,2] <- W7[2,1] <- 0.6
W7[2,3] <- W7[3,2] <- 0.5
W7[4,5] <- W7[5,4] <- 0.7
thresh7 <- c(-1, 0, 1, -0.5, 0.5)  # varied base rates
data7 <- generate_ising_data(n, W7, thresh7, seed = 400)
colnames(data7) <- paste0("V", 1:p)
fit7 <- IsingFit(data7, AND = TRUE, gamma = 0.25, progressbar = FALSE, plot = FALSE)

results$dataset7_unbalanced <- list(
  description = "Unbalanced prevalence with thresholds (p=5, n=300)",
  data = as.list(as.data.frame(data7)),
  n = n, p = p, labels = colnames(data7),
  and_rule = list(weightMatrix = as.matrix(fit7$weiadj), thresholds = as.numeric(fit7$thresholds))
)

# ── Dataset 8: Large sample (p=5, n=1000, chain) ──
set.seed(500)
p <- 5; n <- 1000
W8 <- matrix(0, p, p)
for (i in 1:(p-1)) { W8[i, i+1] <- 0.6; W8[i+1, i] <- 0.6 }
data8 <- generate_ising_data(n, W8, rep(0, p), seed = 500)
colnames(data8) <- paste0("V", 1:p)
fit8 <- IsingFit(data8, AND = TRUE, gamma = 0.25, progressbar = FALSE, plot = FALSE)

results$dataset8_large_n <- list(
  description = "Large sample chain (p=5, n=1000)",
  data = as.list(as.data.frame(data8)),
  n = n, p = p, labels = colnames(data8),
  and_rule = list(weightMatrix = as.matrix(fit8$weiadj), thresholds = as.numeric(fit8$thresholds))
)

# ── Dataset 9: Gamma=0.5 (stricter EBIC) ──
set.seed(600)
p <- 5; n <- 400
W9 <- matrix(0, p, p)
W9[1,2] <- W9[2,1] <- 0.8
W9[2,3] <- W9[3,2] <- 0.6
W9[3,4] <- W9[4,3] <- 0.5
W9[4,5] <- W9[5,4] <- 0.7
W9[1,3] <- W9[3,1] <- 0.3  # weak cross-edge
data9 <- generate_ising_data(n, W9, rep(0, p), seed = 600)
colnames(data9) <- paste0("V", 1:p)
fit9_g025 <- IsingFit(data9, AND = TRUE, gamma = 0.25, progressbar = FALSE, plot = FALSE)
fit9_g050 <- IsingFit(data9, AND = TRUE, gamma = 0.5, progressbar = FALSE, plot = FALSE)

results$dataset9_gamma_comparison <- list(
  description = "Chain+weak cross-edge (p=5, n=400), gamma comparison",
  data = as.list(as.data.frame(data9)),
  n = n, p = p, labels = colnames(data9),
  gamma025 = list(weightMatrix = as.matrix(fit9_g025$weiadj), thresholds = as.numeric(fit9_g025$thresholds)),
  gamma050 = list(weightMatrix = as.matrix(fit9_g050$weiadj), thresholds = as.numeric(fit9_g050$thresholds))
)

write(toJSON(results, digits = 10, auto_unbox = TRUE), "tests/fixtures/ising-extended-ground-truth.json")
cat("Wrote extended IsingFit ground truth\n\n")

# Print all
for (name in names(results)) {
  ds <- results[[name]]
  cat(sprintf("=== %s: %s ===\n", name, ds$description))
  if (!is.null(ds$and_rule)) {
    cat("  AND weights:\n")
    print(round(ds$and_rule$weightMatrix, 4))
    cat("  AND thresholds:", round(ds$and_rule$thresholds, 4), "\n")
  }
  if (!is.null(ds$gamma025)) {
    cat("  gamma=0.25 weights:\n")
    print(round(ds$gamma025$weightMatrix, 4))
    cat("  gamma=0.50 weights:\n")
    print(round(ds$gamma050$weightMatrix, 4))
  }
  cat("\n")
}
