library(shiny)
library('e1071')
library(plyr)
library(rgl)
library(scatterplot3d)
source("./svm_pred.R")

npval = 0.2

AAPL = LOAD_AAPL(np=npval)

chifun <- function(X)
{
  testlist = as.list(t(count(X)[1]))
  names(testlist) = t(count(X)[1])
  return(testlist)
}