### Modified from https://github.com/Azure/Cortana-Intelligence-Gallery-Content/tree/master/Tutorials/Deep-Learning-for-Text-Classification-in-Azure/R

library(mxnet)

alphabet <- c("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
vocab.size <- nchar(alphabet)
feature.len <- 1014
batch_size <- 256

source("vdcnn_model.R")

network <- get_symbol(vocab.size = vocab.size, residual = FALSE, depth = 9, num.output.classes = 2)

data_dir <- "../data"
train.file.input <- file.path(data_dir, "train_amazon.csv")
train.filename <- strsplit(basename(train.file.input), "\\.")[[1]]
train.file.output <-file.path(data_dir, paste0(train.filename[1], "_encoded.", train.filename[2]))
test.file.input <- file.path(data_dir, "val_amazon.csv")
test.filename <- strsplit(basename(test.file.input), "\\.")[[1]]
test.file.output <- file.path(data_dir, paste0(test.filename[1], "_encoded.", test.filename[2]))

source("cnn_util.R")

if (!file.exists(train.file.output)) {
  text.encoder.csv(input.file = train.file.input, output.file = train.file.output,
                   alphabet = alphabet, max.text.lenght = feature.len, shuffle = TRUE)
}

if (!file.exists(test.file.output)) {
  text.encoder.csv(input.file = test.file.input, output.file = test.file.output,
                   alphabet = alphabet, max.text.lenght = feature.len, shuffle = FALSE)
}

# Custom CSVIter --------------------------------------------------------------------
CustomCSVIter <- setRefClass("CustomCSVIter",
                             fields = c("iter", "data.csv", "batch.size", "alphabet", "feature.len"),
                             contains = "Rcpp_MXArrayDataIter",
                             methods = list(initialize = function(iter, data.csv, batch.size, alphabet, feature.len) {
                               csv_iter <- mx.io.CSVIter(data.csv = data.csv, data.shape = feature.len + 1, #=features + label
                                                         batch.size = batch.size)
                              .self$iter <- csv_iter
                              .self$data.csv <- data.csv
                              .self$batch.size <- batch.size
                              .self$alphabet <- alphabet
                              .self$feature.len <- feature.len
                              .self
                            },
                            value = function() {
                              val <- as.array(.self$iter$value()$data)
                              val.y <- val[1, ]
                              val.x <- val[-1, ]
                              val.x <- dict.decoder(data = val.x, alphabet = .self$alphabet,
                                                    feature.len = .self$feature.len, batch.size = .self$batch.size)
                              val.x <- mx.nd.array(val.x)
                              val.y <- mx.nd.array(val.y)
                              list(data = val.x, label = val.y)
                            },
                            iter.next = function() {.self$iter$iter.next()},
                            reset = function() {.self$iter$reset()},
                            num.pad = function() {.self$iter$num.pad()},
                            finalize = function() {.self$iter$finalize()}
                            ))

train.iter <-CustomCSVIter$new(iter = NULL, data.csv = train.file.output,
                               batch.size = batch_size, alphabet = alphabet, feature.len = feature.len)

test.iter <- CustomCSVIter$new(iter = NULL, data.csv = test.file.output,
                               batch.size = batch_size, alphabet = alphabet,feature.len = feature.len)

model <- mx.model.FeedForward.create(symbol = network, X = train.iter, eval.data = test.iter,
                                     ctx = mx.gpu(), num.round = 10, array.batch.size = batch_size,
                                     learning.rate = 0.015, momentum = 0.9, eval.metric = mx.metric.accuracy,
                                     wd = 0.00001, initializer = mx.init.normal(sd = 0.05),
                                     optimizer = "sgd",
                                     batch.end.callback = mx.callback.log.speedometer(batch_size, frequency = 100))
