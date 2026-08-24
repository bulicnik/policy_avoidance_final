library(rstan)
library(bayesplot)
library(loo)


command_args <- commandArgs(trailingOnly = FALSE)
script_arg <- command_args[grep("^--file=", command_args)]
script_path <- normalizePath(sub("^--file=", "", script_arg[1]), winslash = "/")
repo_root <- normalizePath(file.path(dirname(script_path), ".."), winslash = "/")

analysis_args <- commandArgs(trailingOnly = TRUE)
selected_model <- if (length(analysis_args) >= 1) analysis_args[1] else "Rachlin_CW_k"

models <- list(
  Mazur_c = list(file = "indifference_model_Mazur_c.stan", parameters = 2),
  Rachlin_c = list(file = "indifference_model_Rachlin_c.stan", parameters = 3),
  Meyerson_Green_c = list(file = "indifference_model_Meyerson_Green_c.stan", parameters = 3),
  Mazur_CW_k = list(file = "indifference_model_Mazur_CW_k.stan", parameters = 3),
  Rachlin_CW_k = list(file = "indifference_model_Rachlin_CW_k.stan", parameters = 4),
  Meyerson_Green_CW_k = list(file = "indifference_model_Meyerson_Green_CW_k.stan", parameters = 4),
  `Rachlin_CW_k+s` = list(file = "indifference_model_Rachlin_CW_k+s.stan", parameters = 5),
  `Meyerson_Green_CW_k+s` = list(file = "indifference_model_Meyerson_Green_CW_k+s.stan", parameters = 5)
)

if (!selected_model %in% names(models)) {
  stop(paste("Unknown model:", selected_model, "\nChoose one of:", paste(names(models), collapse = ", ")))
}

input_file <- file.path(repo_root, "data", "cmdt_trials.csv")
model_file <- file.path(repo_root, "models", models[[selected_model]]$file)
output_folder <- file.path(repo_root, "results", "model_fits")
dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)

Rinput <- read.csv(input_file)
participant_id <- unlist(Rinput["pid"])
num_participants <- length(unique(participant_id))
trials_per_participant <- as.integer(nrow(Rinput) / num_participants)

data <- list(
  N = nrow(Rinput),
  P = num_participants,
  pid = participant_id,
  A = unlist(Rinput["Death"]),
  D = unlist(Rinput["Employ"]),
  Y = unlist(Rinput["Resp"]),
  u = unlist(Rinput["Between"]),
  w = unlist(Rinput["Social"])
)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

fit <- stan(
  file = model_file,
  data = data,
  iter = 5000,
  chains = 4,
  warmup = 500,
  cores = 4,
  seed = 42
)

summary_stats <- summary(fit)$summary
participant_log_lik <- rstan::extract(fit, "participant_log_lik")$participant_log_lik
mean_participant_log_lik <- apply(participant_log_lik, 2, mean)

k <- models[[selected_model]]$parameters
n <- trials_per_participant
AICs <- 2 * k - 2 * mean_participant_log_lik
BICs <- log(n) * k - 2 * mean_participant_log_lik

participant_results <- data.frame(
  Participant = 1:length(AICs),
  AIC = AICs,
  BIC = BICs
)

write.csv(
  participant_results,
  file.path(output_folder, paste0(selected_model, "_participant_information_criteria.csv")),
  row.names = FALSE
)
write.csv(
  summary_stats,
  file.path(output_folder, paste0(selected_model, "_posterior_summary.csv")),
  row.names = FALSE
)

print(summary(fit))
print(summary_stats[, c("n_eff", "Rhat")])
