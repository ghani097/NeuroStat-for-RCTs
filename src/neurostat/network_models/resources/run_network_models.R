args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7) {
  stop(paste(
    "usage: run_network_models.R INPUT_CSV JOBS_CSV LEVELS_CSV",
    "BETWEEN_CSV WITHIN_CSV SETTINGS_TSV OUTPUT_DIR"
  ))
}
input_path <- args[[1]]
jobs_path <- args[[2]]
levels_path <- args[[3]]
between_path <- args[[4]]
within_path <- args[[5]]
settings_path <- args[[6]]
output_dir <- args[[7]]

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})
options(contrasts = c("contr.sum", "contr.poly"))
emmeans::emm_options(lmer.df = "satterthwaite")

read_csv <- function(path) {
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}
input <- read_csv(input_path)
jobs <- read_csv(jobs_path)
levels <- read_csv(levels_path)
between_spec <- read_csv(between_path)
within_spec <- read_csv(within_path)
settings_table <- read.delim(
  settings_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  colClasses = "character"
)
if (!identical(names(settings_table), c("name", "value"))) {
  stop("settings TSV must have exactly name and value columns")
}
settings <- setNames(settings_table$value, settings_table$name)
required_settings <- c(
  "reml", "optimizer", "maxfun", "residuals", "raw_scale", "winsorized",
  "winsor_robust_z", "all_fit", "influence", "leave_one_out"
)
if (!all(required_settings %in% names(settings))) {
  stop("settings TSV is missing required values")
}
as_flag <- function(name) identical(tolower(settings[[name]]), "true")

empty_frame <- function(columns) {
  result <- as.data.frame(setNames(replicate(length(columns), character(0), simplify = FALSE), columns))
  result
}
append_frame <- function(target, value) {
  value <- as.data.frame(value, stringsAsFactors = FALSE)
  if (nrow(target) == 0) {
    return(value)
  }
  rbind(target, value)
}
write_result <- function(value, filename, columns) {
  if (nrow(value) == 0) {
    value <- empty_frame(columns)
  }
  value <- value[, columns, drop = FALSE]
  write.csv(value, file.path(output_dir, filename), row.names = FALSE, quote = TRUE, na = "")
}
job_prefix <- c("analysis_family", "condition", "band", "network")

fixed_columns <- c(job_prefix, "response", "term", "estimate", "std_error", "df", "statistic", "p_value")
anova_columns <- c(job_prefix, "term", "num_df", "den_df", "f_value", "p_value")
emmeans_columns <- c(
  job_prefix, "arm", "session", "emmean", "std_error", "df", "lower_cl",
  "upper_cl", "raw_emm", "raw_lower_cl", "raw_upper_cl"
)
between_columns <- c(
  job_prefix, "contrast", "session", "numerator_arm", "denominator_arm",
  "estimate", "std_error", "df", "t_ratio", "p_value", "residual_sigma",
  "standardized_estimate"
)
within_columns <- c(
  job_prefix, "contrast", "arm", "numerator_session", "denominator_session",
  "estimate", "std_error", "df", "t_ratio", "p_value", "residual_sigma",
  "standardized_estimate"
)
overview_columns <- c(
  job_prefix, "n_rows", "subjects", "arms", "sessions", "converged",
  "singular", "random_intercept_sd", "residual_sd", "icc", "max_abs_gradient",
  "convergence_message"
)
residual_columns <- c(
  job_prefix, "n", "mean", "sd", "minimum", "maximum", "shapiro_w",
  "shapiro_p", "fitted_residual_correlation"
)
residual_observation_columns <- c(
  job_prefix, "record_id", "subject_id", "arm", "session", "raw_pli", "z_pli",
  "fitted", "residual", "standardized_residual"
)
sensitivity_columns <- c(
  job_prefix, "sensitivity", "term", "estimate", "std_error", "df",
  "statistic", "p_value", "flagged_values"
)
sensitivity_anova_columns <- c(
  job_prefix, "sensitivity", "term", "num_df", "den_df", "f_value", "p_value",
  "singular"
)
sensitivity_between_columns <- c(
  job_prefix, "sensitivity", "contrast", "session", "numerator_arm",
  "denominator_arm", "estimate", "std_error", "df", "t_ratio", "p_value",
  "residual_sigma", "standardized_estimate", "singular", "flagged_values"
)
allfit_columns <- c(
  job_prefix, "optimizer", "success", "singular", "log_likelihood",
  "max_fixed_effect_difference", "message"
)
influence_columns <- c(
  job_prefix, "subject_id", "cooks_distance", "cutoff_4_over_n",
  "influential", "message"
)
loo_columns <- c(
  job_prefix, "excluded_subject_id", "contrast", "session", "numerator_arm",
  "denominator_arm", "estimate", "std_error", "df", "t_ratio", "p_value",
  "full_estimate", "full_t_ratio", "estimate_change", "direction_reversal",
  "singular", "converged", "convergence_message"
)
loo_summary_columns <- c(
  job_prefix, "contrast", "session", "full_estimate", "full_t_ratio",
  "minimum_loo_estimate", "maximum_loo_estimate", "minimum_absolute_loo_t",
  "maximum_absolute_loo_t", "any_direction_reversal", "any_singular"
)

fixed_rows <- empty_frame(fixed_columns)
anova_rows <- empty_frame(anova_columns)
emmeans_rows <- empty_frame(emmeans_columns)
between_rows <- empty_frame(between_columns)
within_rows <- empty_frame(within_columns)
overview_rows <- empty_frame(overview_columns)
residual_rows <- empty_frame(residual_columns)
residual_observation_rows <- empty_frame(residual_observation_columns)
sensitivity_rows <- empty_frame(sensitivity_columns)
sensitivity_anova_rows <- empty_frame(sensitivity_anova_columns)
sensitivity_between_rows <- empty_frame(sensitivity_between_columns)
allfit_rows <- empty_frame(allfit_columns)
influence_rows <- empty_frame(influence_columns)
loo_rows <- empty_frame(loo_columns)
loo_summary_rows <- empty_frame(loo_summary_columns)

prefix_frame <- function(job, rows) {
  cbind(
    data.frame(
      analysis_family = rep(job$analysis_family, rows),
      condition = rep(job$condition, rows),
      band = rep(job$band, rows),
      network = rep(job$network, rows),
      stringsAsFactors = FALSE
    )
  )
}

extract_fixed <- function(fit, job, response, sensitivity = NULL, flagged = NA_integer_) {
  values <- as.data.frame(coef(summary(fit)))
  result <- cbind(
    prefix_frame(job, nrow(values)),
    data.frame(
      response = response,
      term = rownames(values),
      estimate = values[["Estimate"]],
      std_error = values[["Std. Error"]],
      df = values[["df"]],
      statistic = values[["t value"]],
      p_value = values[["Pr(>|t|)"]],
      stringsAsFactors = FALSE
    )
  )
  if (is.null(sensitivity)) {
    return(result)
  }
  data.frame(
    result[, job_prefix, drop = FALSE],
    sensitivity = sensitivity,
    term = result$term,
    estimate = result$estimate,
    std_error = result$std_error,
    df = result$df,
    statistic = result$statistic,
    p_value = result$p_value,
    flagged_values = flagged,
    stringsAsFactors = FALSE
  )
}

model_control <- lme4::lmerControl(
  optimizer = settings[["optimizer"]],
  optCtrl = list(maxfun = as.integer(settings[["maxfun"]]))
)
collapse_messages <- function(value) {
  if (is.null(value) || length(value) == 0) {
    return("")
  }
  paste(unique(unlist(value)), collapse = " | ")
}
is_boundary_singular_message <- function(value) {
  grepl("^boundary \\(singular\\) fit:", trimws(as.character(value)), ignore.case = TRUE)
}
convergence_status <- function(fit) {
  raw_messages <- unique(as.character(unlist(fit@optinfo$conv$lme4$messages)))
  raw_messages <- raw_messages[nzchar(raw_messages)]
  blocking_messages <- raw_messages[
    !vapply(raw_messages, is_boundary_singular_message, logical(1))
  ]
  messages <- collapse_messages(raw_messages)
  code <- fit@optinfo$conv$opt
  opt_success <- !is.null(code) && length(code) > 0 && all(as.integer(code) == 0L)
  list(
    converged = length(blocking_messages) == 0L && opt_success,
    message = paste(
      c(
        if (nchar(messages)) messages else NULL,
        if (!opt_success) sprintf("optimizer convergence code: %s", collapse_messages(code)) else NULL
      ),
      collapse = " | "
    )
  )
}
extract_anova_sensitivity <- function(fit, job, sensitivity) {
  values <- as.data.frame(stats::anova(fit, type = 3, ddf = "Satterthwaite"))
  values <- values[rownames(values) != "(Intercept)", , drop = FALSE]
  cbind(
    prefix_frame(job, nrow(values)),
    data.frame(
      sensitivity = sensitivity,
      term = rownames(values),
      num_df = values[["NumDF"]],
      den_df = values[["DenDF"]],
      f_value = values[["F value"]],
      p_value = values[["Pr(>F)"]],
      singular = lme4::isSingular(fit, tol = 1e-4),
      stringsAsFactors = FALSE
    )
  )
}
extract_between_sensitivity <- function(fit, job, job_between, sensitivity, flagged) {
  result <- empty_frame(sensitivity_between_columns)
  arm_levels <- levels(model.frame(fit)$arm)
  for (contrast_index in seq_len(nrow(job_between))) {
    spec <- job_between[contrast_index, , drop = FALSE]
    weights <- setNames(rep(0, length(arm_levels)), arm_levels)
    weights[[spec$numerator_arm]] <- 1
    weights[[spec$denominator_arm]] <- -1
    grid <- emmeans::emmeans(fit, ~ arm | session, at = list(session = spec$session))
    value <- summary(
      emmeans::contrast(grid, method = setNames(list(weights), spec$name), adjust = "none"),
      infer = c(TRUE, TRUE)
    )
    result <- append_frame(
      result,
      cbind(
        prefix_frame(job, nrow(value)),
        data.frame(
          sensitivity = sensitivity,
          contrast = spec$name,
          session = spec$session,
          numerator_arm = spec$numerator_arm,
          denominator_arm = spec$denominator_arm,
          estimate = value$estimate,
          std_error = value$SE,
          df = value$df,
          t_ratio = value$t.ratio,
          p_value = value$p.value,
          residual_sigma = stats::sigma(fit),
          standardized_estimate = value$estimate / stats::sigma(fit),
          singular = lme4::isSingular(fit, tol = 1e-4),
          flagged_values = flagged,
          stringsAsFactors = FALSE
        )
      )
    )
  }
  result
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

for (job_index in seq_len(nrow(jobs))) {
  job <- jobs[job_index, , drop = FALSE]
  data <- input[
    input$analysis_family == job$analysis_family &
      input$condition == job$condition &
      input$band == job$band &
      input$network == job$network,
    ,
    drop = FALSE
  ]
  arm_levels <- levels[
    levels$analysis_family == job$analysis_family & levels$dimension == "arm",
    ,
    drop = FALSE
  ]
  arm_levels <- arm_levels$value[order(as.integer(arm_levels$position))]
  session_levels <- levels[
    levels$analysis_family == job$analysis_family & levels$dimension == "session",
    ,
    drop = FALSE
  ]
  session_levels <- session_levels$value[order(as.integer(session_levels$position))]
  if (length(arm_levels) != 2 || length(session_levels) < 2) {
    stop(sprintf("job %s/%s/%s has invalid factor levels", job$analysis_family, job$band, job$network))
  }
  data$subject_id <- factor(data$subject_id)
  data$arm <- factor(data$arm, levels = arm_levels)
  data$session <- factor(data$session, levels = session_levels)
  data$z_pli <- as.numeric(data$z_pli)
  data$raw_pli <- as.numeric(data$raw_pli)
  if (anyNA(data[, c("subject_id", "arm", "session", "z_pli", "raw_pli")])) {
    stop(sprintf("job %s/%s/%s contains missing or invalid model data", job$analysis_family, job$band, job$network))
  }
  cell_counts <- table(data$arm, data$session)
  if (any(cell_counts == 0)) {
    stop(sprintf("job %s/%s/%s has an empty arm/session cell", job$analysis_family, job$band, job$network))
  }

  fit <- lmerTest::lmer(
    z_pli ~ arm * session + (1 | subject_id),
    data = data,
    REML = as_flag("reml"),
    control = model_control
  )
  fixed_rows <- append_frame(fixed_rows, extract_fixed(fit, job, "z_pli"))

  av <- as.data.frame(stats::anova(fit, type = 3, ddf = "Satterthwaite"))
  av <- av[rownames(av) != "(Intercept)", , drop = FALSE]
  av_result <- cbind(
    prefix_frame(job, nrow(av)),
    data.frame(
      term = rownames(av),
      num_df = av[["NumDF"]],
      den_df = av[["DenDF"]],
      f_value = av[["F value"]],
      p_value = av[["Pr(>F)"]],
      stringsAsFactors = FALSE
    )
  )
  anova_rows <- append_frame(anova_rows, av_result)

  emm <- summary(emmeans::emmeans(fit, ~ arm * session), infer = c(TRUE, TRUE), adjust = "none")
  emm_result <- cbind(
    prefix_frame(job, nrow(emm)),
    data.frame(
      arm = as.character(emm$arm),
      session = as.character(emm$session),
      emmean = emm$emmean,
      std_error = emm$SE,
      df = emm$df,
      lower_cl = emm$lower.CL,
      upper_cl = emm$upper.CL,
      raw_emm = tanh(emm$emmean),
      raw_lower_cl = tanh(emm$lower.CL),
      raw_upper_cl = tanh(emm$upper.CL),
      stringsAsFactors = FALSE
    )
  )
  emmeans_rows <- append_frame(emmeans_rows, emm_result)

  job_between <- between_spec[between_spec$analysis_family == job$analysis_family, , drop = FALSE]
  for (contrast_index in seq_len(nrow(job_between))) {
    spec <- job_between[contrast_index, , drop = FALSE]
    weights <- setNames(rep(0, length(arm_levels)), arm_levels)
    weights[[spec$numerator_arm]] <- 1
    weights[[spec$denominator_arm]] <- -1
    grid <- emmeans::emmeans(fit, ~ arm | session, at = list(session = spec$session))
    contrast_result <- summary(
      emmeans::contrast(grid, method = setNames(list(weights), spec$name), adjust = "none"),
      infer = c(TRUE, TRUE)
    )
    between_rows <- append_frame(
      between_rows,
      cbind(
        prefix_frame(job, nrow(contrast_result)),
        data.frame(
          contrast = spec$name,
          session = spec$session,
          numerator_arm = spec$numerator_arm,
          denominator_arm = spec$denominator_arm,
          estimate = contrast_result$estimate,
          std_error = contrast_result$SE,
          df = contrast_result$df,
          t_ratio = contrast_result$t.ratio,
          p_value = contrast_result$p.value,
          residual_sigma = stats::sigma(fit),
          standardized_estimate = contrast_result$estimate / stats::sigma(fit),
          stringsAsFactors = FALSE
        )
      )
    )
  }

  job_within <- within_spec[within_spec$analysis_family == job$analysis_family, , drop = FALSE]
  for (contrast_index in seq_len(nrow(job_within))) {
    spec <- job_within[contrast_index, , drop = FALSE]
    weights <- setNames(rep(0, length(session_levels)), session_levels)
    weights[[spec$numerator_session]] <- 1
    weights[[spec$denominator_session]] <- -1
    grid <- emmeans::emmeans(fit, ~ session | arm, at = list(arm = spec$arm))
    contrast_result <- summary(
      emmeans::contrast(grid, method = setNames(list(weights), spec$name), adjust = "none"),
      infer = c(TRUE, TRUE)
    )
    within_rows <- append_frame(
      within_rows,
      cbind(
        prefix_frame(job, nrow(contrast_result)),
        data.frame(
          contrast = spec$name,
          arm = spec$arm,
          numerator_session = spec$numerator_session,
          denominator_session = spec$denominator_session,
          estimate = contrast_result$estimate,
          std_error = contrast_result$SE,
          df = contrast_result$df,
          t_ratio = contrast_result$t.ratio,
          p_value = contrast_result$p.value,
          residual_sigma = stats::sigma(fit),
          standardized_estimate = contrast_result$estimate / stats::sigma(fit),
          stringsAsFactors = FALSE
        )
      )
    )
  }

  convergence <- convergence_status(fit)
  vc <- as.data.frame(lme4::VarCorr(fit))
  random_sd <- vc$sdcor[vc$grp == "subject_id"][[1]]
  residual_sd <- stats::sigma(fit)
  gradient <- fit@optinfo$derivs$gradient
  max_gradient <- if (is.null(gradient)) NA_real_ else max(abs(gradient))
  overview_rows <- append_frame(
    overview_rows,
    cbind(
      prefix_frame(job, 1),
      data.frame(
        n_rows = nrow(data),
        subjects = length(unique(data$subject_id)),
        arms = length(arm_levels),
        sessions = length(session_levels),
        converged = convergence$converged,
        singular = lme4::isSingular(fit, tol = 1e-4),
        random_intercept_sd = random_sd,
        residual_sd = residual_sd,
        icc = random_sd^2 / (random_sd^2 + residual_sd^2),
        max_abs_gradient = max_gradient,
        convergence_message = convergence$message,
        stringsAsFactors = FALSE
      )
    )
  )

  if (as_flag("residuals")) {
    residual_values <- stats::residuals(fit)
    fitted_values <- stats::fitted(fit)
    residual_observation_rows <- append_frame(
      residual_observation_rows,
      cbind(
        prefix_frame(job, nrow(data)),
        data.frame(
          record_id = data$record_id,
          subject_id = as.character(data$subject_id),
          arm = as.character(data$arm),
          session = as.character(data$session),
          raw_pli = data$raw_pli,
          z_pli = data$z_pli,
          fitted = fitted_values,
          residual = residual_values,
          standardized_residual = residual_values / stats::sigma(fit),
          stringsAsFactors = FALSE
        )
      )
    )
    shapiro <- if (length(residual_values) >= 3 && length(residual_values) <= 5000) {
      stats::shapiro.test(residual_values)
    } else {
      NULL
    }
    residual_rows <- append_frame(
      residual_rows,
      cbind(
        prefix_frame(job, 1),
        data.frame(
          n = length(residual_values),
          mean = mean(residual_values),
          sd = stats::sd(residual_values),
          minimum = min(residual_values),
          maximum = max(residual_values),
          shapiro_w = if (is.null(shapiro)) NA_real_ else unname(shapiro$statistic),
          shapiro_p = if (is.null(shapiro)) NA_real_ else shapiro$p.value,
          fitted_residual_correlation = stats::cor(fitted_values, residual_values),
          stringsAsFactors = FALSE
        )
      )
    )
  }

  if (as_flag("raw_scale")) {
    raw_fit <- lmerTest::lmer(
      raw_pli ~ arm * session + (1 | subject_id),
      data = data,
      REML = as_flag("reml"),
      control = model_control
    )
    sensitivity_rows <- append_frame(
      sensitivity_rows,
      extract_fixed(raw_fit, job, "raw_pli", "raw_scale", 0L)
    )
    sensitivity_anova_rows <- append_frame(
      sensitivity_anova_rows,
      extract_anova_sensitivity(raw_fit, job, "raw_scale")
    )
    sensitivity_between_rows <- append_frame(
      sensitivity_between_rows,
      extract_between_sensitivity(raw_fit, job, job_between, "raw_scale", 0L)
    )
  }

  if (as_flag("winsorized")) {
    threshold <- as.numeric(settings[["winsor_robust_z"]])
    winsor <- data$raw_pli
    flagged <- rep(FALSE, nrow(data))
    cells <- interaction(data$arm, data$session, drop = TRUE)
    for (cell in levels(cells)) {
      indexes <- which(cells == cell)
      values <- data$raw_pli[indexes]
      center <- stats::median(values)
      scale <- stats::mad(values, center = center, constant = 1.4826)
      if (!is.finite(scale) || scale <= 0) {
        scale <- stats::IQR(values) / 1.349
      }
      if (!is.finite(scale) || scale <= 0) {
        scale <- stats::sd(values)
      }
      if (is.finite(scale) && scale > 0) {
        lower <- center - threshold * scale
        upper <- center + threshold * scale
        flagged[indexes] <- abs((values - center) / scale) > threshold
        winsor[indexes] <- pmax(lower, pmin(upper, values))
      }
    }
    epsilon <- 1 / (2 * as.numeric(data$sample_count))
    winsor_adjusted <- ifelse(
      winsor == 0,
      epsilon,
      ifelse(winsor == 1, 1 - epsilon, winsor)
    )
    if (any(!is.finite(winsor_adjusted)) || any(winsor_adjusted <= 0 | winsor_adjusted >= 1)) {
      stop(sprintf(
        "winsorized raw PLI boundary transform failed for %s/%s/%s",
        job$analysis_family, job$band, job$network
      ))
    }
    data$raw_pli_winsor <- winsor
    data$z_pli_winsor <- atanh(winsor_adjusted)
    winsor_fit <- lmerTest::lmer(
      z_pli_winsor ~ arm * session + (1 | subject_id),
      data = data,
      REML = as_flag("reml"),
      control = model_control
    )
    sensitivity_rows <- append_frame(
      sensitivity_rows,
      extract_fixed(
        winsor_fit,
        job,
        "z_pli_winsor",
        sprintf("cellwise_raw_pli_robust_winsor_z_%g", threshold),
        sum(flagged)
      )
    )
    sensitivity_anova_rows <- append_frame(
      sensitivity_anova_rows,
      extract_anova_sensitivity(
        winsor_fit,
        job,
        sprintf("cellwise_raw_pli_robust_winsor_z_%g", threshold)
      )
    )
    sensitivity_between_rows <- append_frame(
      sensitivity_between_rows,
      extract_between_sensitivity(
        winsor_fit,
        job,
        job_between,
        sprintf("cellwise_raw_pli_robust_winsor_z_%g", threshold),
        sum(flagged)
      )
    )
  }

  if (as_flag("all_fit")) {
    assign(".neurostat_allfit_data", data, envir = .GlobalEnv)
    alternatives <- tryCatch(
      {
        # allFit rewrites the stored model call. Keep both the data symbol and
        # lmerControl call resolvable while those optimizer-specific refits run.
        allfit_seed <- lme4::lmer(
          z_pli ~ arm * session + (1 | subject_id),
          data = .neurostat_allfit_data,
          REML = as_flag("reml"),
          control = lme4::lmerControl(
            optimizer = settings[["optimizer"]],
            optCtrl = list(maxfun = as.integer(settings[["maxfun"]]))
          )
        )
        suppressWarnings(lme4::allFit(allfit_seed, verbose = FALSE, maxfun = 100000))
      },
      error = function(error) error,
      finally = rm(".neurostat_allfit_data", envir = .GlobalEnv)
    )
    if (inherits(alternatives, "error")) {
      allfit_rows <- append_frame(
        allfit_rows,
        cbind(
          prefix_frame(job, 1),
          data.frame(
            optimizer = "allFit",
            success = FALSE,
            singular = NA,
            log_likelihood = NA,
            max_fixed_effect_difference = NA,
            message = conditionMessage(alternatives),
            stringsAsFactors = FALSE
          )
        )
      )
    } else {
      reference <- lme4::fixef(fit)
      for (optimizer_name in names(alternatives)) {
        alternative <- alternatives[[optimizer_name]]
        ok <- inherits(alternative, "merMod")
        message <- if (ok) {
          collapse_messages(alternative@optinfo$conv$lme4$messages)
        } else {
          collapse_messages(alternative)
        }
        allfit_rows <- append_frame(
          allfit_rows,
          cbind(
            prefix_frame(job, 1),
            data.frame(
              optimizer = optimizer_name,
              success = ok,
              singular = if (ok) lme4::isSingular(alternative, tol = 1e-4) else NA,
              log_likelihood = if (ok) as.numeric(stats::logLik(alternative)) else NA,
              max_fixed_effect_difference = if (ok) {
                max(abs(lme4::fixef(alternative) - reference))
              } else {
                NA
              },
              message = message,
              stringsAsFactors = FALSE
            )
          )
        )
      }
    }
  }

  if (as_flag("influence")) {
    if (!requireNamespace("influence.ME", quietly = TRUE)) {
      stop("requested influence diagnostic requires the influence.ME package")
    }
    influence_object <- tryCatch(
      influence.ME::influence(fit, group = "subject_id"),
      error = function(error) error
    )
    cutoff <- 4 / nlevels(data$subject_id)
    if (inherits(influence_object, "error")) {
      influence_rows <- append_frame(
        influence_rows,
        cbind(
          prefix_frame(job, 1),
          data.frame(
            subject_id = NA,
            cooks_distance = NA,
            cutoff_4_over_n = cutoff,
            influential = NA,
            message = conditionMessage(influence_object),
            stringsAsFactors = FALSE
          )
        )
      )
    } else {
      cook <- stats::cooks.distance(influence_object)
      ids <- names(cook)
      if (is.null(ids) || length(ids) != length(cook)) {
        ids <- levels(data$subject_id)
      }
      influence_rows <- append_frame(
        influence_rows,
        cbind(
          prefix_frame(job, length(cook)),
          data.frame(
            subject_id = ids,
            cooks_distance = as.numeric(cook),
            cutoff_4_over_n = cutoff,
            influential = as.numeric(cook) > cutoff,
            message = "",
            stringsAsFactors = FALSE
          )
        )
      )
    }
  }

  if (as_flag("leave_one_out")) {
    full_between <- between_rows[
      between_rows$analysis_family == job$analysis_family &
        between_rows$band == job$band &
        between_rows$network == job$network,
      ,
      drop = FALSE
    ]
    for (subject in levels(data$subject_id)) {
      reduced <- droplevels(data[data$subject_id != subject, , drop = FALSE])
      reduced_fit <- tryCatch(
        lmerTest::lmer(
          z_pli ~ arm * session + (1 | subject_id),
          data = reduced,
          REML = as_flag("reml"),
          control = model_control
        ),
        error = function(error) stop(sprintf(
          "requested leave-one-out fit failed after excluding %s for %s/%s/%s: %s",
          subject, job$analysis_family, job$band, job$network, conditionMessage(error)
        ))
      )
      reduced_between <- extract_between_sensitivity(
        reduced_fit, job, job_between, "leave_one_out", 0L
      )
      reduced_convergence <- convergence_status(reduced_fit)
      for (contrast_index in seq_len(nrow(reduced_between))) {
        value <- reduced_between[contrast_index, , drop = FALSE]
        reference <- full_between[
          full_between$contrast == value$contrast &
            full_between$session == value$session,
          ,
          drop = FALSE
        ]
        if (nrow(reference) != 1) {
          stop("leave-one-out could not resolve the full planned contrast")
        }
        loo_rows <- append_frame(
          loo_rows,
          cbind(
            prefix_frame(job, 1),
            data.frame(
              excluded_subject_id = subject,
              contrast = value$contrast,
              session = value$session,
              numerator_arm = value$numerator_arm,
              denominator_arm = value$denominator_arm,
              estimate = value$estimate,
              std_error = value$std_error,
              df = value$df,
              t_ratio = value$t_ratio,
              p_value = value$p_value,
              full_estimate = reference$estimate,
              full_t_ratio = reference$t_ratio,
              estimate_change = value$estimate - reference$estimate,
              direction_reversal = sign(value$estimate) != sign(reference$estimate),
              singular = lme4::isSingular(reduced_fit, tol = 1e-4),
              converged = reduced_convergence$converged,
              convergence_message = reduced_convergence$message,
              stringsAsFactors = FALSE
            )
          )
        )
      }
    }
    job_loo <- loo_rows[
      loo_rows$analysis_family == job$analysis_family &
        loo_rows$band == job$band &
        loo_rows$network == job$network,
      ,
      drop = FALSE
    ]
    for (contrast_index in seq_len(nrow(job_between))) {
      contrast_name <- job_between$name[[contrast_index]]
      contrast_session <- job_between$session[[contrast_index]]
      values <- job_loo[
        job_loo$contrast == contrast_name & job_loo$session == contrast_session,
        ,
        drop = FALSE
      ]
      loo_summary_rows <- append_frame(
        loo_summary_rows,
        cbind(
          prefix_frame(job, 1),
          data.frame(
            contrast = contrast_name,
            session = values$session[[1]],
            full_estimate = values$full_estimate[[1]],
            full_t_ratio = values$full_t_ratio[[1]],
            minimum_loo_estimate = min(values$estimate),
            maximum_loo_estimate = max(values$estimate),
            minimum_absolute_loo_t = min(abs(values$t_ratio)),
            maximum_absolute_loo_t = max(abs(values$t_ratio)),
            any_direction_reversal = any(values$direction_reversal),
            any_singular = any(values$singular),
            stringsAsFactors = FALSE
          )
        )
      )
    }
  }
}

write_result(fixed_rows, "fixed_effects.csv", fixed_columns)
write_result(anova_rows, "anova_type3.csv", anova_columns)
write_result(emmeans_rows, "estimated_marginal_means.csv", emmeans_columns)
write_result(between_rows, "between_arm_contrasts.csv", between_columns)
write_result(within_rows, "within_arm_contrasts.csv", within_columns)
write_result(overview_rows, "model_overview.csv", overview_columns)
write_result(residual_rows, "residual_diagnostics.csv", residual_columns)
write_result(
  residual_observation_rows,
  "residual_observations.csv",
  residual_observation_columns
)
write_result(sensitivity_rows, "sensitivity_models.csv", sensitivity_columns)
write_result(sensitivity_anova_rows, "sensitivity_anova_type3.csv", sensitivity_anova_columns)
write_result(
  sensitivity_between_rows,
  "sensitivity_between_arm_contrasts.csv",
  sensitivity_between_columns
)
write_result(allfit_rows, "allfit_diagnostics.csv", allfit_columns)
write_result(influence_rows, "influence_diagnostics.csv", influence_columns)
write_result(loo_rows, "leave_one_out.csv", loo_columns)
write_result(loo_summary_rows, "leave_one_out_summary.csv", loo_summary_columns)
