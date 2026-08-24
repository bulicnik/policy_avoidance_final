args <- commandArgs(trailingOnly = TRUE)

input_path  <- args[1]
output_path <- args[2]
form_str   <- args[3]

form <- as.formula(form_str)

library(lme4)
library(dplyr)

df <- read.csv(input_path)

fit <- glmer(
  form,
  data = df,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa")
)

coefs <- summary(fit)$coefficients


alpha <- 0.05
zcrit <- qnorm(1 - alpha/2)

beta     <- coefs[, "Estimate"]
se       <- coefs[, "Std. Error"]
ci_low   <- beta - zcrit * se
ci_high  <- beta + zcrit * se

# Odds ratios and CI
OR      <- exp(beta)
OR_low  <- exp(ci_low)
OR_high <- exp(ci_high)

# Combine output
out <- data.frame(
  Term      = rownames(coefs),
  Estimate  = beta,
  SE        = se,
  z         = coefs[, "z value"],
  p         = coefs[, "Pr(>|z|)"],
  CI_low    = ci_low,
  CI_high   = ci_high,
  OR        = OR,
  OR_low    = OR_low,
  OR_high   = OR_high,
  row.names = NULL
)

write.csv(out, output_path, row.names = FALSE)


