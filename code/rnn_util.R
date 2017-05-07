
make.dict <- function(text, max.vocab = 10000) {
  text <- strsplit(text, '')
  dic <- list()
  idx <- 1
  for (c in text[[1]]) {
    if (!(c %in% names(dic))) {
      dic[[c]] <- idx
      idx <- idx + 1
    }
  }
  if (length(dic) == max.vocab - 1)
    dic[["UNKNOWN"]] <- idx
  cat(paste0("Total unique char: ", length(dic), "\n"))
  return (dic)
}

make.data <- function(file.path, seq.len = 32, max.vocab = 10000, dic = NULL) {
  fi <- file(file.path, "r")
  text <- paste(readLines(fi), collapse = "\n")
  close(fi)
  
  if (is.null(dic))
    dic <- make.dict(text, max.vocab)
  lookup.table <- list()
  for (c in names(dic)) {
    idx <- dic[[c]]
    lookup.table[[idx]] <- c
  }
  
  char.lst <- strsplit(text, '')[[1]]
  num.seq <- as.integer(length(char.lst) / seq.len)
  char.lst <- char.lst[1:(num.seq * seq.len)]
  data <- array(0, dim = c(seq.len, num.seq))
  idx <- 1
  for (i in 1:num.seq) {
    for (j in 1:seq.len) {
      if (char.lst[idx] %in% names(dic))
        data[j, i] <- dic[[char.lst[idx]]] - 1
      else {
        data[j, i] <- dic[["UNKNOWN"]] - 1
      }
      idx <- idx + 1
    }
  }
  return (list(
    data = data,
    dic = dic,
    lookup.table = lookup.table
  ))
}

drop.tail <- function(X, batch.size) {
  shape <- dim(X)
  nstep <- as.integer(shape[2] / batch.size)
  return (X[, 1:(nstep * batch.size)])
}

get.label <- function(X) {
  label <- array(0, dim = dim(X))
  d <- dim(X)[1]
  w <- dim(X)[2]
  for (i in 0:(w - 1)) {
    for (j in 1:d) {
      label[i * d + j] <- X[(i * d + j) %% (w * d) + 1]
    }
  }
  return (label)
}


cdf <- function(weights) {
  total <- sum(weights)
  result <- c()
  cumsum <- 0
  for (w in weights) {
    cumsum <- cumsum + w
    result <- c(result, cumsum / total)
  }
  return (result)
}

search.val <- function(cdf, x) {
  l <- 1
  r <- length(cdf)
  while (l <= r) {
    m <- as.integer((l + r) / 2)
    if (cdf[m] < x) {
      l <- m + 1
    } else {
      r <- m - 1
    }
  }
  return (l)
}

choice <- function(weights) {
  cdf.vals <- cdf(as.array(weights))
  x <- runif(1)
  idx <- search.val(cdf.vals, x)
  return (idx)
}

make.output <- function(prob, sample = FALSE) {
  if (!sample) {
    idx <- which.max(as.array(prob))
  }
  else {
    idx <- choice(prob)
  }
  return (idx)
  
}
