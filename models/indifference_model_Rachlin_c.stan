data {
  int<lower=0> N;                 // Total number of trials
  int<lower=0> P;                 // Number of participants
  int<lower=1, upper=P> pid[N];   // Participant index for each trial
  vector[N] A;                    // Lives saved in each trial
  vector[N] D;                    // Change in unemployment in each trial
  int<lower=0, upper=1> Y[N];     // Binary responses: 1 for 'yes', 0 for 'no'
  int<lower=0, upper=1> u[N];     // Between Variable 0 means lives saved 1 means reduced deaths
}

parameters {
  vector<lower=0, upper=600>[P] k;  // Participant-specific discount factors, uniformly distributed between 0 and 1
  vector<lower=0.01, upper=2>[P] s;  // Participant-specific discount factors, uniformly distributed between 0 and 1
  real<lower=0, upper=7> c;
}

model {
  // Priors
  for (p in 1:P) {
    k[p] ~ uniform(0, 600);         // Uniform prior for individual k values
    s[p] ~ uniform(0.01, 2);         // Uniform prior for individual s values
    c ~ uniform(0, 7);
  }

  // Model the likelihood of each response
  for (i in 1:N) {
    real utility = A[i] / (1 + k[pid[i]] * D[i]^s[pid[i]]) - c;
    Y[i] ~ bernoulli_logit(utility);  // Logit model incorporating individual utility
  }
}

generated quantities {
  vector[P] participant_log_lik = rep_vector(0, P);
  for (i in 1:N) {
    real utility = A[i] / (1 + k[pid[i]] * D[i]^s[pid[i]]) - c;
    participant_log_lik[pid[i]] += bernoulli_logit_lpmf(Y[i] | utility); // Sum log likelihood by participant
  }
}

