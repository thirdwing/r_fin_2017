require(mxnet)

### the network parameters ###
batch.size = 32
seq.len = 32
num.hidden = 64
num.embed = 32
num.lstm.layer = 3
num.round = 10
learning.rate = 0.1
wd = 0.00001
clip_gradient = 1
update.period = 1

### load util functions ###
source("rnn_util.R")

### read the training data ###
ret <- make.data("../data/lstm_demo_input.txt", seq.len = seq.len)

X <- ret$data
dic <- ret$dic
lookup.table <- ret$lookup.table

vocab <- length(dic)

shape <- dim(X)
train.val.fraction <- 0.9
size <- shape[2]

### training and validation ###

X.train.data <- X[, 1:as.integer(size * train.val.fraction)]
X.val.data <- X[,-(1:as.integer(size * train.val.fraction))]
X.train.data <- drop.tail(X.train.data, batch.size)
X.val.data <- drop.tail(X.val.data, batch.size)

X.train.label <- get.label(X.train.data)
X.val.label <- get.label(X.val.data)

X.train <- list(data = X.train.data, label = X.train.label)
X.val <- list(data = X.val.data, label = X.val.label)

### lstm model ###

model <- mx.lstm(
  X.train,
  X.val,
  ctx = mx.cpu(), ## change into mx.gpu() if you want to use GPU
  num.round = num.round,
  update.period = update.period,
  num.lstm.layer = num.lstm.layer,
  seq.len = seq.len,
  num.hidden = num.hidden,
  num.embed = num.embed,
  num.label = vocab,
  batch.size = batch.size,
  input.size = vocab,
  initializer = mx.init.uniform(0.1),
  learning.rate = learning.rate,
  wd = wd,
  clip_gradient = clip_gradient
)


### model inference ###

infer.model <- mx.lstm.inference(
  num.lstm.layer = num.lstm.layer,
  input.size = vocab,
  num.hidden = num.hidden,
  num.embed = num.embed,
  num.label = vocab,
  arg.params = model$arg.params,
  ctx = mx.gpu()
)

### generate the output ###
### you can try to use different start and seq.len ###

start <- 'b'
seq.len <- 200
random.sample <- TRUE

last.id <- dic[[start]]
out <- "b"
for (i in (1:(seq.len - 1))) {
  input <- c(last.id - 1)
  ret <- mx.lstm.forward(infer.model, input, FALSE)
  infer.model <- ret$model
  prob <- ret$prob
  last.id <- make.output(prob, random.sample)
  out <- paste0(out, lookup.table[[last.id]])
}

print(out)
