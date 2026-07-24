args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("usage: preflight.R CONTRACT_TSV OUTPUT_TSV STRICT_RENV LOCKFILE")
}
contract_path <- args[[1]]
output_path <- args[[2]]
strict_renv <- identical(tolower(args[[3]]), "true")
lockfile <- args[[4]]

contract <- read.delim(
  contract_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  colClasses = "character"
)
if (!identical(names(contract), c("package", "comparison", "version")) || nrow(contract) == 0) {
  stop("invalid dependency contract header or empty contract")
}
if (anyDuplicated(contract$package) || any(!contract$comparison %in% c(">=", "=="))) {
  stop("invalid dependency contract package/comparison rows")
}

reported <- list()
for (index in seq_len(nrow(contract))) {
  package <- contract$package[[index]]
  required <- contract$version[[index]]
  comparison <- contract$comparison[[index]]
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("required R package '%s' is not installed in .libPaths(): %s", package, paste(.libPaths(), collapse = "; ")))
  }
  installed <- as.character(utils::packageVersion(package))
  if (comparison == ">=" && utils::packageVersion(package) < package_version(required)) {
    stop(sprintf("R package '%s' is %s but dependency contract requires >= %s", package, installed, required))
  }
  if (comparison == "==" && utils::packageVersion(package) != package_version(required)) {
    stop(sprintf("R package '%s' is %s but dependency contract requires exactly %s", package, installed, required))
  }
  reported[[length(reported) + 1]] <- data.frame(
    kind = "package", name = package, value = installed, stringsAsFactors = FALSE
  )
}

if (strict_renv) {
  if (!nzchar(lockfile) || !file.exists(lockfile)) {
    stop("strict renv checking requires an existing explicit lockfile")
  }
  if (!requireNamespace("renv", quietly = TRUE)) {
    stop("strict renv checking was requested but the 'renv' package is not installed")
  }
  lock <- renv::lockfile_read(lockfile)
  for (index in seq_len(nrow(contract))) {
    package <- contract$package[[index]]
    locked <- lock$Packages[[package]]
    if (is.null(locked) || is.null(locked$Version)) {
      stop(sprintf("renv lockfile does not declare required package '%s'", package))
    }
    if (as.character(utils::packageVersion(package)) != as.character(locked$Version)) {
      stop(sprintf(
        "installed R package '%s' (%s) differs from renv lockfile (%s)",
        package, as.character(utils::packageVersion(package)), as.character(locked$Version)
      ))
    }
  }
}

runtime <- data.frame(
  kind = "runtime",
  name = "R",
  value = R.version.string,
  stringsAsFactors = FALSE
)
libraries <- data.frame(
  kind = "library",
  name = as.character(seq_along(.libPaths())),
  value = .libPaths(),
  stringsAsFactors = FALSE
)
result <- do.call(rbind, c(list(runtime, libraries), reported))
write.table(
  result,
  output_path,
  sep = "\t",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE,
  fileEncoding = "UTF-8"
)
