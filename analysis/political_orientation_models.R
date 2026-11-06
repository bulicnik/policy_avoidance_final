library(lme4)
library(lmerTest)

script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
root <- dirname(dirname(normalizePath(script)))
output_folder <- file.path(root, "results", "political_orientation")

survey <- read.csv(file.path(root, "data", "survey_responses_deidentified.csv"),
                   fileEncoding = "UTF-8-BOM")
required_columns <- c("ID", "analysis_included", "political_orientation")
stopifnot(all(required_columns %in% names(survey)),
          !anyNA(survey$ID), !anyNA(survey$analysis_included),
          all(survey$analysis_included %in% c(0, 1)))
parameters <- unique(read.csv(file.path(root, "data", "behavioral_task_trials.csv"))[
  c("ID", "k_social", "k_no_social", "s")
])

stopifnot(!anyDuplicated(survey$ID), !anyDuplicated(parameters$ID))

survey <- survey[survey$analysis_included == 1,
                 c("ID", "political_orientation")]
stopifnot(all(survey$ID %in% parameters$ID))
data <- merge(survey, parameters, by = "ID")
data <- data[complete.cases(data), ]
stopifnot(nrow(data) > 2, all(data$political_orientation %in% 1:5),
          sd(data$political_orientation) > 0)

# Center at neutral (3); slopes describe a one-step shift toward conservatism.
data$political_orientation_c <- data$political_orientation - 3

long <- reshape(
  data, varying = list(c("k_no_social", "k_social")), v.names = "k",
  timevar = "condition", times = c("no_social", "social"), direction = "long"
)
# Match the manuscript: standardize k across both conditions and s across people.
long$k <- as.numeric(scale(long$k))
data$s <- as.numeric(scale(data$s))
long$condition <- factor(long$condition, levels = c("no_social", "social"))
long$ID <- factor(long$ID)
# lmerTest adds Satterthwaite tests to the lme4 fit.
policy_fit <- as_lmerModLmerTest(lme4::lmer(
  k ~ political_orientation_c * condition + (1 | ID), data = long
))
# Sensitivity has one value per participant, so no random intercept is needed.
sensitivity_fit <- lm(s ~ political_orientation_c, data = data)

contrasts <- list(
  baseline_policy_avoidance = c(0, 1, 0, 0),
  social_policy_avoidance = c(0, 1, 0, 1),
  social_minus_baseline = c(0, 0, 0, 1)
)
policy_results <- do.call(rbind, lapply(names(contrasts), function(outcome) {
  test <- contest1D(policy_fit, contrasts[[outcome]], confint = TRUE)
  data.frame(
    outcome = outcome, estimate = test$Estimate, se = test$`Std. Error`,
    df = test$df, t_value = test$`t value`, p_value = test$`Pr(>|t|)`,
    l95 = test$lower, u95 = test$upper, n_participants = nrow(data)
  )
}))
sensitivity_test <- coef(summary(sensitivity_fit))["political_orientation_c", ]
sensitivity_ci <- confint(sensitivity_fit)["political_orientation_c", ]
results <- rbind(policy_results, data.frame(
  outcome = "cost_sensitivity", estimate = sensitivity_test["Estimate"],
  se = sensitivity_test["Std. Error"], df = df.residual(sensitivity_fit),
  t_value = sensitivity_test["t value"], p_value = sensitivity_test["Pr(>|t|)"],
  l95 = sensitivity_ci[1], u95 = sensitivity_ci[2], n_participants = nrow(data)
))

if (isSingular(policy_fit)) warning("Policy-avoidance model has a singular fit.")
dir.create(output_folder, recursive = TRUE, showWarnings = FALSE)
write.csv(results, file.path(output_folder, "parameter_associations.csv"),
          row.names = FALSE)
capture.output(
  summary(policy_fit), summary(sensitivity_fit), sessionInfo(),
  file = file.path(output_folder, "model_summary.txt")
)
print(results, row.names = FALSE)
