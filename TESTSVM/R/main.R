
LOAD_AAPL = function(np=0.2)
{
  AAPL = read.csv('./Youtubelist.csv')
  AAPL = AAPL[,-c(1,2,3,10)]#Range to process

  for(i in 1:7)
    AAPL = AAPL[!is.na(AAPL[,i]),]

  #AAPL = AAPL[sample(nrow(AAPL),3000),]#Sampling
  #AAPL = AAPL[order(AAPL$Box),]

  #AAPL$Youtube.Views = scale(AAPL$Youtube.Views)
  #AAPL$Box = scale(AAPL$Box)
  #AAPL$Budget[is.na(AAPL$Budget)] = mean(AAPL$Budget[!is.na(AAPL$Budget)])
  #AAPL$Youtube.Views = scale(AAPL$Youtube.Views)
  #plot(AAPL$Box)

  AAPL = as.data.frame(AAPL)

  np = ceiling(np * nrow(AAPL))

  #Y = ifelse(AAPL$Box>40000000,1,0)

  AAPL$Y = c(rep(0,length(AAPL$Box)))
  #ranklist = c(-1,1e7L,5e7L,1e7L,2e8L)
  ranklist = c(-1,7e6L,3e7L,7e7L,1e8L)
  ranklist = as.numeric(ranklist)
  for(i in 2:length(ranklist))
  {
    AAPL$Y[AAPL$Box<=ranklist[i] & AAPL$Box>ranklist[i-1]] = i-1
  }
  AAPL$Y[AAPL$Box>ranklist[length(ranklist)]] = length(ranklist)

  AAPL = AAPL[,-1]
  return(AAPL)
}

tune_AAPL = function(AAPL, np=0.2)
{
  np = ceiling(np * nrow(AAPL))
  test.index = sample(1:nrow(AAPL), np)
  #test.index = 1:np

  AAPL.test = AAPL[test.index, ]
  AAPL.train = AAPL[-test.index, ]

  tuned = tune.svm(Y ~ ., data = AAPL.train, gamma = 2^(-7:-5), cost = 2^(2:4))
  return(tuned)
}

test_AAPL = function(AAPL, t_cost=16, t_gamma=0.03125, np=0.2)
{
  np = ceiling(np * nrow(AAPL))

  test.index = sample(1:nrow(AAPL), np)
  #test.index = 1:np

  AAPL.test = AAPL[test.index, ]
  AAPL.train = AAPL[-test.index, ]
  if(nrow(count(AAPL.train$Y))>1)
  {
    svm.model = svm(Y ~ ., data = AAPL.train, kernal='radial', type = 'C-classification', cost = t_cost, gamma = t_gamma)
    #svm.model = readRDS("movie.svmodel")
    svm.pred = predict(svm.model, AAPL.test[, -7])
    table.svm.test = table(pred = svm.pred, true = AAPL.test[, 7])
    correct.svm = sum(diag(table.svm.test) / sum(table.svm.test)) * 100
    result = cbind(AAPL.test, svm.pred)
    result = result[order(result$Y),]#Sorting
    return(list(correct.svm,result,svm.model))
  }else{
    print("Assert: There is only one label.")
    return(-1)
  }
}
