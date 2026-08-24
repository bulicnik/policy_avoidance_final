from rpy2.robjects import conversion, default_converter, pandas2ri, r
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
output_folder = ROOT / "results" / "efa"
survey_input = ROOT / "data" / "private" / "covid_survey_responses.csv"
behavioral_input = ROOT / "data" / "behavioral_task_trials.csv"

if not survey_input.exists():
    raise FileNotFoundError(
        f"Missing private survey input: {survey_input}\n"
        "See data/private/README.md before running this analysis."
    )

debug = False
demograghic_raw = pd.read_csv(survey_input)
demograghic_raw = demograghic_raw.drop(columns=["openness", "conscientiousness","agreeableness", "extroversion", 'emotional_stability'])
demograghic_filtered = demograghic_raw.iloc[0:124,:]
demograghic_filtered = demograghic_filtered.rename(columns={
    'Q118': 'personal_danger',
    'Q207': 'political_orientation',
    'Q239': 'moral1',
    'Q240': 'moral2',
    'Q241': 'moral3',
    'Q242': 'moral4',
    'Q243': 'moral5',
    'ae1' : 'animal1',
    'ae2' : 'animal2',
    'ae3' : 'animal3',
    'ae4' : 'animal4',
    'a35' : 'human',
    'ae6' : 'natural',
    'ae7' : 'foodprep',
    'ae8' : 'natureinterferance',
    'ae9' : 'manmade',
    'avoid4' : 'peopleavoid1',
    'avoid5' : 'peopleavoid2',
    'avoid7' : 'peopleavoid3',
    'avoid8' : 'peopleavoid4',
    'avoid10' : 'peopleavoid5',
    'Avoid.Nyc':'peopleavoid6',
    'AvoidItaly':'peopleavoid7',
    'ItIm':'peopleavoid8',
    'ItDec':'peopleavoid9',
    'tip1' : 'extroversion1',
    'tip6_rev' : 'extroversion2',
    'tip3' : 'conscientiousness1',
    'tip8_rev' : 'conscientiousness2',
    'tip5' : 'openness1',
    'tip10_rev' : 'openness2',
    'tip7' : 'agreeableness1',
    'tip2_rev' : 'agreeableness2',
    'tip9' : 'emotional_stability1',
    'tip4_rev' : 'emotional_stability2',
})



full_df = pd.read_csv(behavioral_input)
filter_df = full_df[['k_social','k_no_social','s','ID', 'IUSScore']].copy()
filter_df.drop_duplicates(inplace=True)
filter_df.reset_index(inplace=True, drop=True)

filter_df['k_social_diff'] = filter_df['k_social'] - filter_df['k_no_social']

dataframes_to_merge = [filter_df]
for dataframe in dataframes_to_merge:
    full_df_with_demo = demograghic_filtered.merge(dataframe, on='ID', how='left')

full_df_with_demo_performance = full_df_with_demo

likert_sections = ['heal', 'd', 'ec', 'Soc', 'Prep', 'stigma','avoid', 'risk',
                   'moral', 'animal', 'people']
model_df = full_df_with_demo_performance[[
    'age', 'rural', 'suburban', 'urban', 'sex',
    'ID', 'k_social', 'k_no_social', 's',
]].copy()


orphan_items = [ 'personal_danger']
Beliefs = ['hoax', 'human','natural', 'manmade', 'foodprep', 'natureinterferance',]
big5 = ['openness', 'agreeableness', 'extroversion', 'conscientiousness',
                'emotional_stability',]
problematic_item_mapping = {
                            'animal1': {np.float64(8.0): 1, np.float64(9.0): 2, np.float64(10.0): 3, np.float64(11.0): 4, np.float64(12.0): 5, np.float64(13.0): 6, np.float64(14.0): 7},
                            'animal2': {np.float64(8.0): 1, np.float64(9.0): 2, np.float64(10.0): 3, np.float64(11.0): 4, np.float64(12.0): 5, np.float64(13.0): 6, np.float64(14.0): 7},
                            'animal3': {np.float64(8.0): 1, np.float64(9.0): 2, np.float64(10.0): 3, np.float64(11.0): 4, np.float64(12.0): 5, np.float64(13.0): 6, np.float64(14.0): 7},
                            'animal4': {np.float64(8.0): 1, np.float64(9.0): 2, np.float64(10.0): 3, np.float64(11.0): 4, np.float64(12.0): 5, np.float64(13.0): 6, np.float64(14.0): 7},
                            'd1': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd2': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd3': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd4': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd5': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd6': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd7': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd8': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd9': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd10': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd11': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd12': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd13': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5},
                            'd14': {np.float64(1.0): 1, np.float64(3.0): 2, np.float64(4.0): 3, np.float64(5.0): 4, np.float64(6.0): 5}}

def run_Dimension_Reduction(likert_sections, model_df, behavioral_df, orphan_items, model_type):
    scaler = StandardScaler()

    if model_type == "sectional_efa":
        first_order_loadings = pd.DataFrame()
        for section in likert_sections:
            print(section)
            section_cols = [col for col in behavioral_df.columns if
                            col.startswith(section) and not col.startswith('diet')]
            section_df = behavioral_df[section_cols].dropna()


            ro.globalenv["section_items"] = py_to_r(section_df)

            ro.r(f'''
            library(psych)
            library(nFactors)

            
            # Polychoric correlation (Likert items)
            poly_result <- polychoric(section_items)
            poly_cor <- poly_result$rho

            # Parallel analysis
            pa <- fa.parallel(poly_cor, n.obs = nrow(section_items), fm = "ml", fa = "fa", plot = FALSE)
            num_factors <- pa$nfact

            fa_result_obj <- fa(poly_cor,
                                nfactors = num_factors,
                                fm = "ml",
                                rotate = "oblimin")

            section_scores <- factor.scores(section_items,
                                            fa_result_obj,
                                            method = "tenBerge")$scores
            # Extract loadings
            first_order_loadings <- as.data.frame(unclass(fa_result_obj$loadings))
            first_order_loadings$Item <- rownames(fa_result_obj$loadings)

            assign("{section}_scores", section_scores)
            assign("{section}_loadings", first_order_loadings)
            ''')

            # Convert scores to pandas
            with conversion.localconverter(ro.default_converter + pandas2ri.converter):
                section_scores = ro.globalenv[f"{section}_scores"]
            score_df = pd.DataFrame(section_scores)
            score_df.columns = [f"{section}_f{i + 1}" for i in range(score_df.shape[1])]
            model_df = pd.concat([model_df.reset_index(drop=True),
                                  score_df.reset_index(drop=True)], axis=1)

            # Convert loadings to pandas and store
            with conversion.localconverter(ro.default_converter + pandas2ri.converter):
                loadings_df = ro.globalenv[f"{section}_loadings"]
            # Make Item the index
            loadings_df = loadings_df.set_index("Item")
            first_order_loadings = pd.concat([first_order_loadings, loadings_df], axis=1)

        # Standardize orphan items
        for orphan in orphan_items:
            if orphan in behavioral_df.columns:
                model_df[f"{orphan}_z"] = scaler.fit_transform(
                    behavioral_df[[orphan]]
                )

        return model_df, first_order_loadings
    elif model_type == "pca":
        loadings = pd.DataFrame()
        section_dfs = []

        for section in likert_sections:
            section_cols = [col for col in behavioral_df.columns if
                            col.startswith(section) and not col.startswith('diet')]
            section_df = behavioral_df[section_cols]
            section_dfs.append(section_df)

        combined_df = pd.concat(section_dfs, axis=1)
        if debug == True:
            print(f"Shape before removing NaN: {combined_df.shape}")
            combined_df = combined_df.dropna()
            print(f"Shape after removing NaN: {combined_df.shape}")

            # ===== ADD THIS BEFORE  =====
            print("\n" + "=" * 70)
            print("DATA BEING SENT TO CORRELATION")
            print("=" * 70)
            print(f"Shape: {combined_df.shape}")
            print(f"Columns: {list(combined_df.columns)}")

            print("\nQuick check of each column:")
            for col in combined_df.columns:
                unique_vals = sorted(combined_df[col].unique())
                max_prop = combined_df[col].value_counts(normalize=True).max()
                std_val = combined_df[col].std()

                warning = ""
                if max_prop > 0.90:
                    warning += f" {max_prop:.1%} in one category"
                if std_val < 0.5:
                    warning += f" SD={std_val:.3f}"
                if len(unique_vals) < 3:
                    warning += f" Only {len(unique_vals)} values"

                print(f"  {col}: {unique_vals} {warning}")

        # Now check and remove problematic items
        problems_in_combined = identify_problematic_items(combined_df, threshold=0.85)

        if problems_in_combined:
            if debug == True:
                print("\n PROBLEMATIC ITEMS FOUND IN COMBINED DATA:")
                for item, issues in problems_in_combined.items():
                    print(f"  {item}: {', '.join(issues)}")

            items_to_remove = list(problems_in_combined.keys())
            combined_df = combined_df.drop(columns=items_to_remove)
            if debug == True:
                print(f"\n Removed {len(items_to_remove)} items.")
                print(f" Continuing with {combined_df.shape[1]} items: {list(combined_df.columns)}\n")
        else:
            if debug == True:
                items_to_remove = list(problems_in_combined.keys())
                print(f"\n Removed {len(items_to_remove)} items.")
                print(f" Continuing with {combined_df.shape[1]} items: {list(combined_df.columns)}\n")

            print("=" * 70 + "\n")
            # ===== END CHECK =====

        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["items"] = combined_df

        ro.r(f'''
                library(psych)
                library(nFactors)
                
                cor_matrix <- cor(items, use = "pairwise.complete.obs")
                
                # Parallel analysis for PCA
                pa <- fa.parallel(
                cor_matrix,
                n.obs = nrow(items),
                fa = "pc",
                plot = TRUE
                )
                
                num_components <- pa$ncomp
                
                pca_result_obj <- principal(
                cor_matrix,
                nfactors = num_components,
                rotate = "oblimin"
                )
                
                scores <- factor.scores(
                  items,
                  pca_result_obj,
                  method = "tenBerge"
                )$scores
                
                scores <- as.data.frame(scores)
                colnames(scores) <- paste0("PCA_Factor_", seq_len(ncol(scores)))
                
                assign("scores", scores)
                loadings <- as.data.frame(unclass(pca_result_obj$loadings))
                loadings$Item <- rownames(pca_result_obj$loadings)
                
                assign("loadings", loadings)
                ''')

        # Convert scores to pandas
        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            scores = ro.globalenv[f"scores"]
        score_df = pd.DataFrame(scores)

        model_df = pd.concat([model_df.reset_index(drop=True),
                              score_df.reset_index(drop=True)], axis=1)

        # Convert loadings to pandas and store
        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            loadings_df = ro.globalenv[f"loadings"]
        # Make Item the index
        loadings_df = loadings_df.set_index("Item")
        loadings = pd.concat([loadings, loadings_df], axis=1)

        # Standardize orphan items
        for orphan in orphan_items:
            if orphan in behavioral_df.columns:
                model_df[f"{orphan}_z"] = scaler.fit_transform(
                    behavioral_df[[orphan]]
                )

        return model_df, loadings
    elif model_type == "full_efa":
        loadings = pd.DataFrame()
        section_dfs = []

        for section in likert_sections:
            section_cols = [col for col in behavioral_df.columns if
                            col.startswith(section) and not col.startswith('diet')]
            section_df = behavioral_df[section_cols]
            section_dfs.append(section_df)

        combined_df = pd.concat(section_dfs, axis=1)

        # Standardize orphan items
        for orphan in orphan_items:
            if orphan in behavioral_df.columns:
                combined_df[f"{orphan}"] = scaler.fit_transform(
                    behavioral_df[[orphan]]
                )

        if debug == True:
            print(f"Shape before removing NaN: {combined_df.shape}")
            combined_df = combined_df.dropna()
            print(f"Shape after removing NaN: {combined_df.shape}")

            print("\n" + "=" * 70)
            print("DATA BEING SENT TO CORRELATION")
            print("=" * 70)
            print(f"Shape: {combined_df.shape}")
            print(f"Columns: {list(combined_df.columns)}")

            print("\nQuick check of each column:")
            for col in combined_df.columns:
                unique_vals = sorted(combined_df[col].unique())
                max_prop = combined_df[col].value_counts(normalize=True).max()
                std_val = combined_df[col].std()

                warning = ""
                if max_prop > 0.90:
                    warning += f" {max_prop:.1%} in one category"
                if std_val < 0.5:
                    warning += f" SD={std_val:.3f}"
                if len(unique_vals) < 3:
                    warning += f" Only {len(unique_vals)} values"

                print(f"  {col}: {unique_vals} {warning}")

        problems_in_combined = identify_problematic_items(combined_df, threshold=0.85)

        if problems_in_combined:
            if debug == True:
                print("\n PROBLEMATIC ITEMS FOUND IN COMBINED DATA:")
                for item, issues in problems_in_combined.items():
                    print(f"  {item}: {', '.join(issues)}")

            items_to_remove = list(problems_in_combined.keys())
            combined_df = combined_df.drop(columns=items_to_remove)
            if debug == True:
                print(f"\n✓ Removed {len(items_to_remove)} items.")
                print(f"✓ Continuing with {combined_df.shape[1]} items: {list(combined_df.columns)}\n")
        else:
            if debug == True:
                items_to_remove = list(problems_in_combined.keys())
                print(f"\n✓ Removed {len(items_to_remove)} items.")
                print(f"✓ Continuing with {combined_df.shape[1]} items: {list(combined_df.columns)}\n")

            print("=" * 70 + "\n")

        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            ro.globalenv["items"] = combined_df

        
        ro.r(r'''
        library(psych)
        library(nFactors)

        # If you want fit indices that are most defensible for ordinal Likert data,
        # consider replacing cor() with polychoric(items)$rho.
        cor_matrix <- cor(items, use = "pairwise.complete.obs")
        alpha_result <- psych::alpha(items)

        fa_parallel <- fa.parallel(
            cor_matrix,
            n.obs = nrow(items),
            fa = "fa",
            fm = "ml",
            plot = FALSE
        )

        # Retention is determined by PA
        num_factors <- fa_parallel$nfact

        # Table of observed vs simulated eigenvalues for reporting
        parallel_results <- data.frame(
            Factor = seq_along(fa_parallel$fa.values),
            Observed = as.numeric(fa_parallel$fa.values),
            Simulated = as.numeric(fa_parallel$fa.sim)
        )
        parallel_results$Retain <- parallel_results$Observed > parallel_results$Simulated

        # Save scree plot with PA comparison
        scree_plot_path <- tempfile(pattern = "efa_scree_", fileext = ".png")
        png(scree_plot_path, width = 1600, height = 1200, res = 200)

        plot(
            parallel_results$Factor,
            parallel_results$Observed,
            type = "b",
            pch = 19,
            xlab = "Factor Number",
            ylab = "Eigenvalue",
            main = "Scree Plot with Parallel Analysis"
        )
        lines(
            parallel_results$Factor,
            parallel_results$Simulated,
            type = "b",
            pch = 17,
            lty = 2
        )
        abline(v = num_factors, lty = 3)

        abline(v = 5, lty = 3, col = "red")

        legend(
                "topright",
                legend = c(
                "Observed eigenvalues",
                "Simulated eigenvalues",
                paste0("PA retained ", num_factors, " factors"),
                paste0("Elbow Point at 5th LV")
            ),
            lty = c(1, 2, 3, 3),
            pch = c(19, 17, NA, NA),
            col = c("black", "black","black", "red"),
            bty = "n"
        )


        dev.off()

        fa_result_obj <- fa(
            cor_matrix,
            nfactors = 5, #use num_factors if you want to simply apply the PA result. We keep 5 here because it is a visual elbow,
            n.obs = nrow(items),
            fm = "ml",
            rotate = "oblimin"
        )

        # Scores
        scores <- factor.scores(
          items,
          fa_result_obj,
          method = "tenBerge"
        )$scores
        scores <- as.data.frame(scores)
        colnames(scores) <- paste0("EFA_Factor_", seq_len(ncol(scores)))
        assign("scores", scores)

        # Loadings
        loadings <- as.data.frame(unclass(fa_result_obj$loadings))
        loadings$Item <- rownames(fa_result_obj$loadings)
        factor_cols <- colnames(loadings)[colnames(loadings) != "Item"]
        colnames(loadings)[colnames(loadings) != "Item"] <- paste0("EFA_Factor_", seq_len(length(factor_cols)))
        assign("loadings", loadings)

        # Variance-accounted table
        vaccounted <- as.data.frame(fa_result_obj$Vaccounted)
        vaccounted$Metric <- rownames(fa_result_obj$Vaccounted)
        colnames(vaccounted)[colnames(vaccounted) != "Metric"] <- paste0("EFA_Factor_", seq_len(ncol(vaccounted) - 1))
        assign("vaccounted", vaccounted)

        # RMSEA can be a named vector; extract safely
        rmsea_point <- NA_real_
        rmsea_low <- NA_real_
        rmsea_high <- NA_real_

        if (!is.null(fa_result_obj$RMSEA)) {
            rmsea_vec <- fa_result_obj$RMSEA

            if (!is.null(names(rmsea_vec))) {
                if ("RMSEA" %in% names(rmsea_vec)) {
                    rmsea_point <- unname(rmsea_vec["RMSEA"])
                } else {
                    rmsea_point <- unname(rmsea_vec[1])
                }

                low_idx <- grep("lower", names(rmsea_vec), ignore.case = TRUE)
                high_idx <- grep("upper", names(rmsea_vec), ignore.case = TRUE)

                if (length(low_idx) > 0) rmsea_low <- unname(rmsea_vec[low_idx[1]])
                if (length(high_idx) > 0) rmsea_high <- unname(rmsea_vec[high_idx[1]])
            } else {
                if (length(rmsea_vec) >= 1) rmsea_point <- unname(rmsea_vec[1])
                if (length(rmsea_vec) >= 2) rmsea_low <- unname(rmsea_vec[2])
                if (length(rmsea_vec) >= 3) rmsea_high <- unname(rmsea_vec[3])
            }
        }

        # One-row fit + retention summary
        efa_fit <- data.frame(
            retained_n_factors = num_factors,
            n_obs = nrow(items),
            n_items = ncol(items),
            pa_last_retained_observed = parallel_results$Observed[num_factors],
            pa_last_retained_simulated = parallel_results$Simulated[num_factors],
            pa_next_observed = if (num_factors < nrow(parallel_results)) parallel_results$Observed[num_factors + 1] else NA_real_,
            pa_next_simulated = if (num_factors < nrow(parallel_results)) parallel_results$Simulated[num_factors + 1] else NA_real_,
            cumulative_var = fa_result_obj$Vaccounted["Cumulative Var", ncol(fa_result_obj$Vaccounted)],
            cumulative_common_var = fa_result_obj$Vaccounted["Cumulative Proportion", ncol(fa_result_obj$Vaccounted)],
            RMSEA = rmsea_point,
            RMSEA_low90 = rmsea_low,
            RMSEA_high90 = rmsea_high,
            TLI = if (!is.null(fa_result_obj$TLI)) fa_result_obj$TLI else NA_real_,
            RMSR = if (!is.null(fa_result_obj$rms)) fa_result_obj$rms else NA_real_
        )

        assign("parallel_results", parallel_results)
        assign("efa_fit", efa_fit)
        assign("scree_plot_path", scree_plot_path)
        ''')

        alpha_result = ro.globalenv["alpha_result"]   # outside conversion if possible
        alphas_total_r = alpha_result.rx2("total")

        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            alphas_total = ro.conversion.rpy2py(alphas_total_r)

        print(alphas_total)
        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            scores = ro.globalenv["scores"]
            vaccounted_df = ro.globalenv["vaccounted"]
            efa_fit_df = ro.globalenv["efa_fit"]
            parallel_results_df = ro.globalenv["parallel_results"]
            loadings_df = ro.globalenv["loadings"]

        score_df = pd.DataFrame(scores)

        model_df = pd.concat([model_df.reset_index(drop=True),
                      score_df.reset_index(drop=True)], axis=1)

        # Convert loadings to pandas
        loadings_df = loadings_df.set_index("Item")
        loadings = pd.concat([loadings, loadings_df], axis=1)


        vaccounted_df = pd.DataFrame(vaccounted_df).set_index("Metric")
        efa_fit_df = pd.DataFrame(efa_fit_df)
        parallel_results_df = pd.DataFrame(parallel_results_df)

        scree_plot_path = str(ro.globalenv["scree_plot_path"][0])

        outdir = Path(output_folder)
        outdir.mkdir(parents=True, exist_ok=True)

        parallel_results_df.to_csv(outdir / "parallel_analysis.csv", index=False)
        efa_fit_df.to_csv(outdir / "fit_summary.csv", index=False)
        shutil.copy2(scree_plot_path, outdir / "scree_plot.png")

        return model_df, loadings, vaccounted_df, efa_fit_df

    else:
        raise ValueError("Invalid model_type. Choose 'sectional_efa', 'full_efa', or 'pca'.")


def run_bayesian_fixed_effect_analysis(model_df):

    ro.globalenv['model_data'] = pandas2ri.py2rpy(model_df)

    ro.r('''
    library(brms)

    baseline_fit <- brm(
      formula = k_social + k_no_social ~ (1|ID),
      data = model_data,
      family = gaussian(),
      chains = 4,
      iter = 4000,
      warmup = 1000,
      control = list(adapt_delta = 0.95),
      seed = 42
    )   
    baseline_loo <- loo(baseline_fit)
    ''')

    # You already have factor columns like trust_f1, risk_f1, etc.
    factor_candidates = [col for col in model_df.columns if "_f" in col and model_df[col].notna().any()]

    improving_factors = []
    current_formula = "k_social + k_no_social ~ "
    best_loo = ro.r("baseline_loo")



    ro.globalenv['model_data'] = pandas2ri.py2rpy(model_df)

    for factor in factor_candidates:
        # Build formula with one more factor
        trial_formula = f"{current_formula} + {factor} + (1|ID)"
        ro.globalenv["trial_formula"] = trial_formula

        # Fit and evaluate
        ro.r('''
        trial_fit <- brm(
          formula = as.formula(trial_formula),
          data = model_data,
          family = gaussian(),
          chains = 4, iter = 4000, warmup = 1000,
          control = list(adapt_delta = 0.95),
          seed = 42, silent = TRUE
        )
        trial_loo <- loo(trial_fit)
        ''')

        # Compare LOO scores
        comparison = ro.r("loo_compare(trial_loo, baseline_loo)")
        elpd_diff = comparison[0, 0]  # first column = elpd_diff

        if elpd_diff < -2:  # negative = improvement
            improving_factors.append(factor)
            #current_formula += f" + {factor}"
            best_loo = ro.r("trial_loo")
            ro.r("baseline_loo <- trial_loo")  # update baseline

    final_formula = current_formula + " + (1|ID)"
    ro.globalenv["final_formula"] = final_formula
    ro.r('''
    final_fit <- brm(
      formula = as.formula(final_formula),
      data = model_data,
      family = gaussian(),
      chains = 4, iter = 4000,
      control = list(adapt_delta = 0.95),
      seed = 42
    )
    ''')

    # Extract summary table back into Python
    summary_df = ro.r('as.data.frame(summary(final_fit)$fixed)')

    # Convert to pandas
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        summary_df = ro.conversion.rpy2py(summary_df)

    print(summary_df)

    ro.r('plot(final_fit)')

    ro.r('pp_check(final_fit)')

    posterior_samples = ro.r('as.data.frame(as_draws_df(final_fit))')
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        posterior_samples = ro.conversion.rpy2py(posterior_samples)

    return summary_df, posterior_samples, best_loo


def recode_to_consecutive(df, recode_map=None, columns=None):
    """
    Recode ordinal columns to consecutive integers starting from 1
    Preserves the original ordering but removes gaps.

    Parameters:
    df: DataFrame
    recode_map: dictionary containing all specified columns and the mapping of options to consecutive options.
    columns: list of column names to recode (if None, applies to all numeric columns)

    Returns:
    df_recoded: DataFrame with recoded columns
    recode_map: Dictionary showing the recoding for each column
    """
    df_recoded = df.copy()
    if recode_map is None:
        print("No mapping specified. Automatically mapping. WARNING! Ensure that all possible options are present in the data before recoding.")
        recode_map = {}

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns

        for col in columns:
            # Get unique values in sorted order, excluding NaN
            unique_vals = sorted(df[col].dropna().unique())

            # Check if recoding is needed (gaps or doesn't start at 1)
            if len(unique_vals) > 0:
                expected_vals = list(range(1, len(unique_vals) + 1))

                if unique_vals != expected_vals:
                    # Create mapping from old to new values
                    mapping = {old: new for new, old in enumerate(unique_vals, start=1)}
                    recode_map[col] = mapping

                    # Apply mapping
                    df_recoded[col] = df[col].map(mapping)

                    print(f"{col}: {unique_vals} -> {expected_vals}")

        return df_recoded, recode_map
    else:
        print("Recode mapping specified. Applying mapping.")

        for col, mapping in recode_map.items():
            if col in df_recoded.columns:
                df_recoded[col] = df_recoded[col].map(mapping)
                print(f"Applied mapping to {col}")
            else:
                print(f"Warning: Column '{col}' not found in DataFrame")
        return df_recoded, recode_map


def run_second_order_PCA(model_df, likert_sections, orphan_items):
    first_order_factors = [
        c for c in model_df.columns
        if c.startswith(tuple(likert_sections)) and "_f" in c
    ]
    pca_input_cols = first_order_factors + orphan_items
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["factor_score_df"] = model_df[pca_input_cols]
    ro.r('''
    library(psych)

    # Standardize factor scores
    fs_scaled <- scale(factor_score_df)

    # Parallel analysis for PCA
    pa2 <- fa.parallel(
      fs_scaled,
      fa = "pc",
      n.iter = 100,
      plot = FALSE
    )

    n_pc <- pa2$ncomp
    if (is.null(n_pc) || n_pc < 1) n_pc <- 1

    # PCA with oblique rotation
    pca2 <- principal(
      fs_scaled,
      nfactors = n_pc,
      rotate = "oblimin",
      scores = TRUE
    )

    second_order_scores <- pca2$scores
    second_order_loadings <- pca2$loadings
    ''')
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        so_scores = ro.globalenv["second_order_scores"]
        so_loadings = ro.globalenv["second_order_loadings"]

    so_scores = pd.DataFrame(so_scores)
    so_scores.columns = [f"SO_f{i + 1}" for i in range(so_scores.shape[1])]

    model_df = pd.concat(
        [model_df.reset_index(drop=True),
         so_scores.reset_index(drop=True)],
        axis=1
    )
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        so_scores = ro.globalenv["second_order_scores"]
        so_loadings = ro.globalenv["second_order_loadings"]

    loading_rnames = [
        c for c in model_df.columns
        if c.startswith(tuple(likert_sections + orphan_items))
    ]
    if len(loading_rnames) != len(so_loadings):
        raise ValueError(
            f"Mismatch: {len(loading_rnames)} names for {len(so_loadings)} rows"
        )
    so_loadings = pd.DataFrame(so_loadings)
    so_loadings['Loading'] = loading_rnames
    so_scores = pd.DataFrame(so_scores)
    so_scores.columns = [f"SO_f{i + 1}" for i in range(so_scores.shape[1])]

    model_df = pd.concat(
        [model_df.reset_index(drop=True),
         so_scores.reset_index(drop=True)],
        axis=1
    )
    return model_df, so_scores, so_loadings


def get_significant_latent_factors(dataframe, factor_names, dep_variable= None, chains=4, iter=4000):
    """
    Iterates over factor_names, fits a brms model predicting dep_variable by each factor,
    and returns a list of significant predictors, posterior summaries, and Bayes factors.

    Returns:
    -------
    results: list of dict
        Each dict has keys: 'factor', 'estimate', 'se', 'l95', 'u95', 'BF'
    """

    scaler = StandardScaler()
    dataframe.loc[:,dep_variable] = scaler.fit_transform(dataframe[[dep_variable]])

    brms = importr("brms")
    base = importr("base")

    significant_results = []

    for factor in factor_names:
        print(f"\n=== Fitting model for {factor} predicting {dep_variable} ===\n")

        # Convert pandas to R dataframe
        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            r_df = r["as.data.frame"](dataframe)

        # Create formula
        formula_str = f"{dep_variable} ~ {factor}"
        formula = r(f"as.formula('{formula_str}')")

        fit = brms.brm(
            formula=formula,
            data=r_df,
            chains=chains,
            iter=iter,
            refresh=0
        )

        # Extract fixed effects summary
        fe = r["as.data.frame"](r["fixef"](fit, summary=True))
        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            fe_df = pandas2ri.rpy2py(fe)

        factor_row = fe_df.iloc[1]  # first row after intercept
        cols = list(fe_df.columns)
        estimate = factor_row[cols[0]]
        se = factor_row[cols[1]]
        l95 = factor_row[cols[2]]
        u95 = factor_row[cols[3]]

        # === Get Bayes factor via hypothesis test ===
        if estimate > 0:
            hyp_str = f"{factor} > 0"
        elif estimate <= 0:
            hyp_str = f"{factor} < 0"
        hyp = brms.hypothesis(fit, hyp_str)
        hyp_df = r["as.data.frame"](hyp.rx2("hypothesis"))

        with conversion.localconverter(ro.default_converter + pandas2ri.converter):
            hyp_py = pandas2ri.rpy2py(hyp_df)

        print(hyp_py)
        bf = hyp_py.iloc[0]["Evid.Ratio"]

        print(f"{factor}: Estimate = {estimate:.3f}, 95% CI = [{l95:.3f}, {u95:.3f}], BF = {bf:.2f}")

        if l95 > 0 or u95 < 0:
            significant_results.append({
                'factor': factor,
                'estimate': estimate,
                'se': se,
                'l95': l95,
                'u95': u95,
                'BF': bf
            })

    return significant_results

def get_latent_factors_lm(dataframe, factor_names, dep_variable=None):
    """
    Fits one multiple linear model predicting dep_variable by all factors in factor_names,
    and returns estimates, SE, CI, t-statistic, and p-value for each requested factor.

    Returns
    -------
    results_df : pandas.DataFrame
        Columns: 'factor', 'estimate', 'se', 'l95', 'u95', 't_value', 'p_value'
    """

    scaler = StandardScaler()
    dataframe = dataframe.copy()
    dataframe.loc[:, f"{dep_variable}_z"] = scaler.fit_transform(dataframe[[dep_variable]])

    stats = importr("stats")
    base = importr("base")

    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        r_df = r["as.data.frame"](dataframe)

    r_code = """
    get_results <- function(df, factors, dep_var) {

    factors <- as.character(unlist(factors))
    dep_var <- as.character(dep_var)

    valid_factors <- factors[factors %in% colnames(df)]

    results <- data.frame(
        model = NA_character_,
        factor = factors,
        estimate = NA_real_,
        se = NA_real_,
        l95 = NA_real_,
        u95 = NA_real_,
        t_value = NA_real_,
        p_value = NA_real_,
        stringsAsFactors = FALSE
    )

    if (length(valid_factors) == 0) {
        return(results)
    }

    form <- as.formula(paste(dep_var, "~", paste(valid_factors, collapse = " + ")))
    model_string <- paste(deparse(form), collapse = "")

    fit <- try(lm(form, data = df), silent = TRUE)

    if (inherits(fit, "try-error")) {
        results$model <- model_string
        return(results)
    }

    summ <- summary(fit)
    coefs <- summ$coefficients
    ci <- try(confint(fit, level = 0.95), silent = TRUE)

    results$model <- model_string

    for (i in seq_along(factors)) {
        f <- factors[i]

        if (f %in% rownames(coefs)) {
            results$estimate[i] <- coefs[f, "Estimate"]
            results$se[i] <- coefs[f, "Std. Error"]
            results$t_value[i] <- coefs[f, "t value"]
            results$p_value[i] <- coefs[f, "Pr(>|t|)"]

            if (!inherits(ci, "try-error") && f %in% rownames(ci)) {
                results$l95[i] <- ci[f, 1]
                results$u95[i] <- ci[f, 2]
            }
        }
    }

    return(results)
    }
    """

    r(r_code)

    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        results_df = r["get_results"](r_df, factor_names, f"{dep_variable}_z")

    return results_df

def get_latent_factors_lm_per_factor(dataframe, factor_names, dep_variable=None):
    """
    Iterates over factor_names, fits a linear model predicting dep_variable by each factor,
    and returns a list of predictors with estimates, SE, CI, p-value, and t-statistic.

    Returns:
    -------
    results_df : pandas.DataFrame
        Columns: 'factor', 'estimate', 'se', 'l95', 'u95', 't_value', 'p_value'
    """

    scaler = StandardScaler()
    dataframe.loc[:, f"{dep_variable}_z"] = scaler.fit_transform(dataframe[[dep_variable]])


    stats = importr("stats")
    base = importr("base")

    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        r_df = r["as.data.frame"](dataframe)

    r_code = """
    get_results <- function(df, factors, dep_var) {
        results <- data.frame(
            factor = character(),
            estimate = numeric(),
            se = numeric(),
            l95 = numeric(),
            u95 = numeric(),
            t_value = numeric(),
            p_value = numeric(),
            stringsAsFactors = FALSE
        )

        for (f in factors) {
            if (!f %in% colnames(df)) {
                results <- rbind(results, data.frame(
                    factor = f,
                    estimate = NA,
                    se = NA,
                    l95 = NA,
                    u95 = NA,
                    t_value = NA,
                    p_value = NA,
                    stringsAsFactors = FALSE
                ))
                next
            }

            form <- as.formula(paste(dep_var, "~", f))
            fit <- try(lm(form, data=df), silent = TRUE)

            if (inherits(fit, "try-error")) {
                results <- rbind(results, data.frame(
                    factor = f,
                    estimate = NA,
                    se = NA,
                    l95 = NA,
                    u95 = NA,
                    t_value = NA,
                    p_value = NA,
                    stringsAsFactors = FALSE
                ))
                next
            }

            summ <- summary(fit)
            coefs <- summ$coefficients

            if (!f %in% rownames(coefs)) {
                results <- rbind(results, data.frame(
                    factor = f,
                    estimate = NA,
                    se = NA,
                    l95 = NA,
                    u95 = NA,
                    t_value = NA,
                    p_value = NA,
                    stringsAsFactors = FALSE
                ))
                next
            }

            ci <- try(confint(fit, level = 0.95), silent = TRUE)

            results <- rbind(results, data.frame(
                factor = f,
                estimate = coefs[f, "Estimate"],
                se = coefs[f, "Std. Error"],
                l95 = if (!inherits(ci, "try-error")) ci[f, 1] else NA,
                u95 = if (!inherits(ci, "try-error")) ci[f, 2] else NA,
                t_value = coefs[f, "t value"],
                p_value = coefs[f, "Pr(>|t|)"],
                stringsAsFactors = FALSE
            ))
        }
        results
    }
    """

    # Load R function into environment
    r(r_code)

    # Call the R function
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        results_df = r["get_results"](r_df, factor_names, f"{dep_variable}_z")

    return results_df


def fit_factors_long_brms_model(df, factors):
    """
    Fits the given brms model in R using rpy2, returns both fit object and summary table.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have columns: ['ID', 'k_social', 'k_no_social', , 'full_1', 'full_2', 'full_3', 'full_4', 'full_5']

    Returns
    -------
    fit : R object (brmsfit)
        The fitted model in R's memory.
    summary_df : pandas.DataFrame
        Parameter summary table from brms::summary(fit) as a pandas DataFrame.
    """



    # Send pandas DataFrame to R
    ro.globalenv['Rinput1'] = py_to_r(df)

    significant_results = []

    for factor in factors:
        r_code_fit = f"""
            library(tidyverse)
            library(brms)

            Rinput1_long <- Rinput1 %>%
              pivot_longer(cols = c(k_social, k_no_social),
                           names_to = "Condition",
                           values_to = "k") %>%
              mutate(
                k_z = as.numeric(scale(k))
              )

            fit <- brm(
              formula = k_z ~ {factor} * Condition + (1|ID),
              data = Rinput1_long,
              chains = 4, cores = 7, iter = 4000
            )
            fit
        """

        # Run the model in R
        fit = ro.r(r_code_fit)

        # Get the summary table
        summary_df = ro.r("as.data.frame(summary(fit)$fixed)")

        # Ensure it's a pandas DataFrame
        from rpy2.robjects import pandas2ri

        if not isinstance(summary_df, pd.DataFrame):
            summary_df = pandas2ri.rpy2py(summary_df)

        # Look for the interaction term (factor + Condition)
        interaction_rows = summary_df.loc[
            summary_df.index.str.contains(f"{factor}:Condition")
        ]

        if not interaction_rows.empty:
            # Take the first matching interaction row
            interaction_row = interaction_rows.iloc[0]
            estimate = interaction_row['Estimate']
            se = interaction_row['Est.Error']
            l95 = interaction_row['l-95% CI']
            u95 = interaction_row['u-95% CI']

            print(
                f"\nEffect of {factor} × Condition interaction:\n"
                f"  Estimate = {estimate:.3f} (SE = {se:.3f})\n"
                f"  95% CI   = [{l95:.3f}, {u95:.3f}]"
            )

            # Mark significant if CI excludes 0
            if l95 > 0 or u95 < 0:
                significant_results.append({
                    'factor': factor,
                    'term': interaction_rows.index[0],
                    'estimate': estimate,
                    'se': se,
                    'l95': l95,
                    'u95': u95
                })
        else:
            print(f"\nNo interaction term found for {factor}")
    return significant_results


def fit_factors_long_lmer_model(df, factors):
    """
    Fits a model in R (no random effect for ID) using rpy2.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have columns: ['ID', 'k_social', 'k_no_social' , 'full_1', 'full_2', 'full_3', 'full_4', 'full_5']

    Returns
    -------
    significant_results : list of dict
        Significant interaction results.
    """

    # Send pandas DataFrame to R
    ro.globalenv['Rinput1'] = py_to_r(df)

    significant_results = []

    for factor in factors:
        r_code_fit = f"""
            library(tidyverse)

            Rinput1_long <- Rinput1 %>%
              pivot_longer(cols = c(k_social, k_no_social),
                           names_to = "Condition",
                           values_to = "k") %>%
              mutate(
                Condition = ifelse(Condition == "k_social", "social", "no_social"),
                k_z = as.numeric(scale(k))
              )

            library(lmerTest)
            fit <- lmer(
              formula = k_z ~ {factor} * Condition + (1|ID),
              data = Rinput1_long
            )
            fit
        """

        fit = ro.r(r_code_fit)

        # Extract summary and CIs
        r_summary_code = """
        coefs <- as.data.frame(coef(summary(fit)))
        coefs$term <- rownames(coefs)

        ci <- as.data.frame(confint(fit))
        ci$term <- rownames(ci)

        merged <- merge(coefs, ci, by="term", all.x=TRUE)
        merged
        """
        summary_df = ro.r(r_summary_code)

        from rpy2.robjects import pandas2ri

        if not isinstance(summary_df, pd.DataFrame):
            summary_df = pandas2ri.rpy2py(summary_df)

        interaction_rows = summary_df.loc[
            summary_df['term'].str.contains(f"{factor}:Condition")
        ]

        if not interaction_rows.empty:
            interaction_row = interaction_rows.iloc[0]
            estimate = interaction_row['Estimate']
            se = interaction_row['Std. Error']
            tval = interaction_row['t value']
            df_val = interaction_row['df']
            pval = interaction_row['Pr(>|t|)']
            l95 = interaction_row['2.5 %']
            u95 = interaction_row['97.5 %']

            print(
                f"\nEffect of {factor} × Condition interaction:\n"
                f"  Estimate = {estimate:.3f} (SE = {se:.3f})\n"
                f"  t = {tval:.2f}, p = {pval:.3f}\n"
                f"  95% CI   = [{l95:.3f}, {u95:.3f}]"
            )

            if l95 > 0 or u95 < 0:
                significant_results.append({
                    'factor': factor,
                    'term': interaction_rows.index[0],
                    'estimate': estimate,
                    'tval': tval,
                    'df_val': df_val,
                    'pval': pval,
                    'se': se,
                    'l95': l95,
                    'u95': u95
                })
        else:
            print(f"\nNo interaction term found for {factor}")

    return significant_results


def py_to_r(obj):
    with conversion.localconverter(default_converter + pandas2ri.converter):
        return conversion.py2rpy(obj)

def r_to_py(obj):
    with conversion.localconverter(default_converter + pandas2ri.converter):
        return conversion.rpy2py(obj)

def reliability_of_section(section_prefix, df):
    """
    Compute Cronbach's alpha using R's psych::alpha() via rpy2.
    section_prefix: e.g., 'heal', 'risk', 'avoid'
    df: pandas dataframe with your full item set
    """
    cols = [c for c in df.columns if c.startswith(section_prefix)]
    section_df = df[cols]

    if len(cols) < 2:
        print(f"Section '{section_prefix}' has fewer than 2 items.")
        return None

    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["alpha_items"] = section_df
        r("""
        library(psych)
        alpha_result <- psych::alpha(alpha_items)
        """)
        alpha_value = r("alpha_result$total$raw_alpha")[0]

    print(f"Cronbach's alpha for '{section_prefix}' ({len(cols)} items): {alpha_value:.3f}")
    return alpha_value


def identify_problematic_items(items_df, threshold=0.90):
    """Find items causing the polychoric error"""
    problems = {}

    for col in items_df.columns:
        issues = []

        # Check 1: Variance
        if items_df[col].std() < 0.5:
            issues.append(f"Low variance (SD={items_df[col].std():.2f})")

        # Check 2: Extreme distribution
        props = items_df[col].value_counts(normalize=True)
        max_prop = props.max()
        if max_prop > threshold:
            issues.append(f"Extreme distribution ({max_prop:.1%} in one category)")

        # Check 3: Few unique values
        n_unique = items_df[col].nunique()
        if n_unique < 3:
            issues.append(f"Only {n_unique} unique values")

        # Check 4: Missing data
        missing_pct = items_df[col].isnull().mean()
        if missing_pct > 0.3:
            issues.append(f"High missing data ({missing_pct:.1%})")

        if issues:
            problems[col] = issues

    return problems


def drop_problematic_items(items_df, problems):
    if debug == True:
        print("Problematic items:")
        for item, issues in problems.items():
            print(f"\n{item}:")
            for issue in issues:
                print(f"  - {issue}")
    items_to_remove = list(problems.keys())
    if debug == True:
        print(f"Removing {len(items_to_remove)} problematic items: {items_to_remove}\n")
    cleaned_df = items_df.drop(columns=items_to_remove)
    return cleaned_df

def save_list_of_dfs(results, result_names, filepath=None, index = False):
    """
    parameters: 
    results: a DataFrame or a list of DataFrames
    result_names: a list of strings that will be used to name the resulting files
    filepath: where the files will be saved. Default: current working directory
    index: a boolean that deterimines if the index is kept. Default: False/no index
    output: None
    """
    if filepath is None:
        filepath = ""  
    else:
        filepath = str(filepath)
        Path(filepath).mkdir(parents=True, exist_ok=True)
        if not filepath.endswith(("/", "\\")):
            filepath += "/"  

    if isinstance(results, pd.DataFrame):
        results = [results]

    for i, result_name in enumerate(result_names):
        results[i].to_csv(f"{filepath}{result_name}.csv", index=index)
'''
Running the analysis
'''

cols_to_recode = [col for col in full_df_with_demo_performance.columns
               if col.startswith(tuple(likert_sections)) and not col.startswith('diet')]

full_df_with_demo_performance, mapping = recode_to_consecutive(full_df_with_demo_performance,problematic_item_mapping, cols_to_recode)
problems = identify_problematic_items(full_df_with_demo_performance[cols_to_recode])
cleaned_demo_df = drop_problematic_items(full_df_with_demo_performance, problems)

model_df, first_order_loadings, efa_vaccounted, efa_fit = run_Dimension_Reduction(
    likert_sections, model_df, cleaned_demo_df, orphan_items, model_type='full_efa'
)

model_df_clean = model_df.dropna()
#model_df_SO, df_SO, SO_loadings = run_second_order_PCA(model_df_clean,likert_sections, ['personal_danger_z',"IUSScore_z",'political_orientation_z'])
#summary_df, posterior_samples, best_lo = run_bayesian_fixed_effect_analysis(model_df_clean)
dep_variables = ['k_social', 'k_no_social', 's']
save_results = [None] * len(dep_variables)
pca_factor_names = ['PCA_Factor_1', 'PCA_Factor_2', 'PCA_Factor_3','PCA_Factor_4','PCA_Factor_5', 'PCA_Factor_6']
full_efa_factor_names = ['EFA_Factor_1', 'EFA_Factor_2', 'EFA_Factor_3','EFA_Factor_4','EFA_Factor_5',] 
for i, dep_variable in enumerate(dep_variables):
    save_results[i] = get_latent_factors_lm(
        model_df_clean,
        full_efa_factor_names,
        dep_variable=dep_variable
    )

print(save_results, type(save_results))
print(first_order_loadings, type(first_order_loadings))

save_list_of_dfs(
    save_results,
    ['social_policy_avoidance_regression', 'baseline_policy_avoidance_regression', 'cost_sensitivity_regression'],
    filepath=output_folder
)
save_list_of_dfs(first_order_loadings, ['factor_loadings'], filepath=output_folder, index=True)
significant_results = fit_factors_long_lmer_model(model_df_clean, full_efa_factor_names)
if significant_results:
    save_list_of_dfs(pd.DataFrame(significant_results), ['significant_condition_interactions'], filepath=output_folder)
save_list_of_dfs(efa_vaccounted, ['variance_accounted'], filepath=output_folder, index=True)
save_list_of_dfs(efa_fit, ['fit_summary'], filepath=output_folder, index=False)

print(efa_fit)
print(efa_vaccounted)

if debug == True:
    print("Problematic items:")
    for item, issues in problems.items():
        print(f"\n{item}:")
        for issue in issues:
            print(f"  - {issue}")


