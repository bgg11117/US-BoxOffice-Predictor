#rm(list=ls(all=TRUE))
library(e1071)
library(scales)
library(plyr)
library(ggplot2)
library(tidyr)
library(lattice)
library(stringi)
library(quantreg)
library(SparseM)
library(caret)
library(partykit)
library(grid)
library(C50)
library(rpart)
library(ipred)
library(MASS)
library(kernlab)
library(randomForest)
library(vcd)
library(class)
library(gmodels)
library(xgboost)
library(Matrix)
library(dplyr)
library(DT)

Sys.setlocale("LC_TIME", "English")
AAPL = read.csv('./input/Youtubelist.csv')
AAPL <- AAPL %>% filter(Youtube.Views<100000000)
AAPX <- AAPL
AAPB <- AAPL
#--------------feature engineering-------------------
AAPL$Distrubutor[AAPL$Distrubutor=='DreamWorks'] <- 'Paramount'
AAPL$Distrubutor[AAPL$Distrubutor=='Miramax'] <- 'Buena Vista'
AAPL$Distrubutor[AAPL$Distrubutor %in% c('Columbia','Samuel Goldwyn',
                                         'MGM','TriStar')] <- 'Sony / Columbia'
AAPL[grepl('Lions',AAPL$Distrubutor),][8] <- 'Lionsgate'
levels(AAPL$Distrubutor)[levels(AAPL$Distrubutor)=='Unknown'] <- 'NA'
AAPL$Distrubutor[AAPL$Distrubutor=='Unknown'] <- 'NA'

#trick(one class)
levels(AAPL$Distrubutor)[levels(AAPL$Distrubutor)=="IFC"] <- "Independent Studio"
AAPL$Distrubutor[!AAPL$Distrubutor%in% c('Sony / Columbia','Warner Bros.','Fox',
                                         'Universal','Buena Vista','Paramount','Lionsgate',"NA")] <- "Independent Studio"


#fox searchlight , sony
#AAPL[grepl('Sony',AAPL$Distrubutor),][8] <- 'Sony / Columbia'
#AAPL[grepl('Fox',AAPL$Distrubutor),][8] <- 'Fox'

AAPL$MPAA[AAPL$MPAA=='GP'] <- 'PG'
levels(AAPL$MPAA)[levels(AAPL$MPAA)=="Unknown"] <- "NA"
AAPL$MPAA[AAPL$MPAA %in% c('Not Yet Rated','Unrated')] <- 'NA'

levels(AAPL$Genre)[levels(AAPL$Genre)=="Unknown"] <- "NA"

#data processing
AAPL.svm = AAPL[,-c(1:5,7,12)]#Range to process

for(i in 1:7)
{AAPL.svm = AAPL.svm[!is.na(AAPL.svm[,i]),]}
AAPL.svm$Y = c(rep(0,length(AAPL.svm$Box)))
ranklist = c(-1,1e7L,5e7L,1e8L)
ranklist = as.numeric(ranklist)
for(i in 2:length(ranklist))
{
  AAPL.svm$Y[AAPL.svm$Box<=ranklist[i] & AAPL.svm$Box>ranklist[i-1]] = i-1
}
AAPL.svm$Y[AAPL.svm$Box>ranklist[length(ranklist)]] = length(ranklist)

AAPL.svm = AAPL.svm[,-6]
AAPL.svm= as.data.frame(AAPL.svm)
AAPL.svm$Y = as.factor(AAPL.svm$Y)
AAPL.svm$Youtube.Views = scale(AAPL.svm$Youtube.Views)
AAPL.svm$Runtime = scale(AAPL.svm$Runtime)

AAPL.svm[AAPL.svm=="NA"]<-NA
AAPL.svm <- na.omit(AAPL.svm)
AAPL.svm$Genre <- droplevels(AAPL.svm$Genre)
AAPL.svm$MPAA <- droplevels(AAPL.svm$MPAA)
AAPL.svm$Distrubutor <- droplevels(AAPL.svm$Distrubutor)

#movie predict - svm + xgboost
#sparse_matrix for xgboost
set.seed(123)
train_sample <- sample(nrow(AAPL.svm),nrow(AAPL.svm)*0.8)
AAPL.svm_train <- AAPL.svm[train_sample,]
AAPL.svm_test <- AAPL.svm[-train_sample,]

outputvector <- as.numeric((AAPL.svm_train$Y))-1 #xgboost(startfrom 0)

train_sparse_matrix <- sparse.model.matrix(Y~.-1, data = AAPL.svm_train)
test_sparse_matrix <- sparse.model.matrix(Y~.-1, data = AAPL.svm_test )
head(train_sparse_matrix)

#cross-validation to choose the parameters
m=nlevels(AAPL.svm$Y)
param = list("objective" = "multi:softprob",
             "eval_metric" = "mlogloss",
             "num_class" = m)

cv.nround <- 200
cv.nfold <- 10
bst.cv = xgb.cv(param=param, data=train_sparse_matrix, label=outputvector, 
                nfold = cv.nfold, nrounds = cv.nround)
nround <- which(bst.cv$test.mlogloss.mean==min(bst.cv$test.mlogloss.mean))
#select the number of trees with the smallest test mlogloss for model building

bst <- xgboost(data = train_sparse_matrix, label = outputvector,
               param=param, nrounds = nround )
pred <- predict(bst,test_sparse_matrix)

pred = t(matrix(pred,m,length(pred)/m))
pred = levels(AAPL.svm_test$Y)[max.col(pred)]

# confusion matrix
xg_accuracy = table(AAPL.svm_test$Y,pred)
xg_accuracy
xg_ac <- sum(diag(xg_accuracy))/sum(xg_accuracy)
#accuracy = 66%

#xgboost importance(train set or entity?)
#AAPL.sparse.matrix <- sparse.model.matrix(Y~.-1, data = AAPL.svm)(if all)

importance <- xgb.importance(train_sparse_matrix @Dimnames[[2]], model = bst)
head(importance)
xgb.plot.importance(head(importance))

#SVM- evaluating model performance

#tuned = tune.svm(Y ~ ., data = AAPL.svm_train, gamma = 2^(-7:-5), cost = 2^(2:4))
#summary(tuned)
svm.model = svm(x=train_sparse_matrix , y = AAPL.svm_train$Y, kernal='radial', type = 'C-classification'
                , cost = 16, gamma = 0.03125)

#data = AAPL.svm or train_sparse_matrix(with Y)

AAPL.svm_pred = predict(svm.model, test_sparse_matrix)
table(AAPL.svm_pred, AAPL.svm_test$Y)

correction <- AAPL.svm_pred == AAPL.svm_test$Y
prop.table(table(correction)) 
svm_ac <- round(prop.table(table(correction))[[2]],2)
#accuracy = 65%

#----- sparse matrix for no split(use occasionally)
#sparse_matrix <- sparse.model.matrix(Y~.-1, data = AAPL.svm)

#---------if classify only 2 classes
#method 1 : xgb.cv( just change the objective)
#"objective" = "binary:logistic"

#method 2 : no xgb.cv
#outputvector <- AAPL.svm[,c(7)] == '1'
#bst <- xgboost(data = sparse_matrix, label = outputvector, max.depth = 4,
#eta = 1, nthread = 2, nround = 10,objective = "binary:logistic")

#---------------------------- knn

knn_train_sparse <-sparse.model.matrix(Y~.-1, data = AAPL.svm_train)
trainoutputvector <- AAPL.svm_train$Y
knn_test_sparse <- sparse.model.matrix(Y~.-1, data = AAPL.svm_test)
testoutputvector <- AAPL.svm_test$Y

knn.fit <- knn(train=knn_train_sparse, test=knn_test_sparse , cl= trainoutputvector,
               k=10)

CrossTable(x=testoutputvector , y=knn.fit , prop.chisq = FALSE)
prop.table(table(testoutputvector==knn.fit))
knn_ac <- round(prop.table(table(testoutputvector==knn.fit))[[2]],2)
#accuracy 63%

#----------------------------


#bonus
control <- trainControl(method = 'repeatedcv',number=10,repeats = 3)

#Cart
set.seed(300)
fit.cart <- train(Y~., data = AAPL.svm_train , method = 'rpart',trControl = control)
AAPL.cart_pred<- predict(fit.cart,AAPL.svm_test)
table(AAPL.cart_pred, AAPL.svm_test$Y)
correction <- AAPL.cart_pred == AAPL.svm_test$Y
prop.table(table(correction))
cart_ac <- round(prop.table(table(correction))[[2]],2)
#accuracy 60%

#LDA
set.seed(300)
fit.lda <- train(Y~., data = AAPL.svm_train , method = 'lda',trControl = control)
AAPL.lda_pred<- predict(fit.lda,AAPL.svm_test)
table(AAPL.lda_pred, AAPL.svm_test$Y)
correction <- AAPL.lda_pred == AAPL.svm_test$Y
prop.table(table(correction)) 
lda_ac <- round(prop.table(table(correction))[[2]],2)
#accuracy 63% 

#RandomForest
set.seed(300)
fit.rf <- train(Y~., data = AAPL.svm_train , method = 'rf',trControl = control)
AAPL.rf_pred<- predict(fit.rf,AAPL.svm_test)
table(AAPL.rf_pred, AAPL.svm_test$Y)
correction <- AAPL.rf_pred == AAPL.svm_test$Y
prop.table(table(correction)) 
rf_ac <- round(prop.table(table(correction))[[2]],2)
#accuracy 64%

#collect resamples
resampleresults <- resamples(list(CART = fit.cart , LDA = fit.lda , RF = fit.rf))

summary(resampleresults)

#plot accuracy

results <- t(as.data.frame(list(XGboost = xg_ac, SVM= svm_ac, knn= knn_ac,
                                CART = cart_ac ,  RF = rf_ac)))
results <- as.data.frame(results)
colnames(results) <- 'Accuracy'

ggplot(results,aes(reorder(rownames(results),-Accuracy),Accuracy))+
  geom_bar(stat='identity',fill="#009E73",width=0.25)+
  coord_cartesian(ylim = c(0,1))+
  geom_text(data = results,aes(x= rownames(results),y=Accuracy,label=Accuracy,vjust=-1,size=3))+
  ggtitle('ML method accuracy')+xlab('method')+
  theme(panel.background = element_blank(),
        axis.line.x = element_line(color='black',size=0.25),
        axis.line.y = element_line(color='black',size=0.25),
        axis.ticks.x=element_blank(),
        legend.position = 'None')                                                    

ggsave('method.png')


#Accuracy : Xgboost is the best 

#----------------ggplot data analysis---------------------
AAPX <- AAPX[,-c(1,3,12)]
AAPX$Distrubutor[AAPX$Distrubutor=='DreamWorks'] <- 'Paramount'
AAPX$Distrubutor[AAPX$Distrubutor=='Miramax'] <- 'Buena Vista'
AAPX$Distrubutor[AAPX$Distrubutor %in% c('Columbia','Samuel Goldwyn',
                                         'MGM','TriStar')] <- 'Sony / Columbia'
AAPX[grepl('Lions',AAPX$Distrubutor),][6] <- 'Lionsgate'
levels(AAPX$Distrubutor)[levels(AAPX$Distrubutor)=='Unknown'] <- 'NA'
AAPX$Distrubutor[AAPX$Distrubutor=='Unknown'] <- 'NA'

#trick(one class)

AAPX$MPAA[AAPX$MPAA=='GP'] <- 'PG'
levels(AAPX$MPAA)[levels(AAPX$MPAA)=="Unknown"] <- "NA"
AAPX$MPAA[AAPX$MPAA %in% c('Not Yet Rated','Unrated')] <- 'NA'

levels(AAPX$Genre)[levels(AAPX$Genre)=="Unknown"] <- "NA"
AAPX[AAPX=="NA"]<-NA
AAPX<- na.omit(AAPX)
AAPX$Genre <- droplevels(AAPX$Genre)
AAPX$MPAA <- droplevels(AAPX$MPAA)
AAPX$Distrubutor <- droplevels(AAPX$Distrubutor)
any(is.na(AAPX))
AAPX$Release.Date<- as.Date(AAPX$Release.Date, format = "%Y/%m/%d")
#AAPX$Month <- format(AAPX$Release.Date,"%b")
AAPX$Month <- factor(AAPX$Month, levels = c("Jan", "Feb","Mar","Apr",'May','Jun',
                                            'Jul','Aug','Sep','Oct','Nov','Dec'))


#Box per MPAA
MPAA <- AAPX %>% group_by(MPAA)%>% summarise(Avg_Box=round(mean(Box),2)) %>%
  arrange(desc(Avg_Box))
datatable(MPAA)

ggplot(MPAA,aes(reorder(MPAA,Avg_Box),Avg_Box,group=1,fill=MPAA))+geom_bar(stat='identity')+
  xlab("MPAA")+ylab("Avg_Box")+ggtitle("Avg_Box of MPAA from 1980-2016")+
  theme(axis.text.x = element_text())+ coord_flip()

#Box per Genre
Genre  <- AAPX %>% group_by(Genre)%>%summarise(Avg_Box=round(mean(Box),2))%>%
  arrange(desc(Avg_Box))
datatable(Genre)

ggplot(Genre ,aes(reorder(Genre,Avg_Box),Avg_Box,group=1,fill=Genre))+geom_bar(stat='identity')+
  xlab("Genre")+ylab("Avg_box") +theme(legend.position='None')+coord_flip()

Genre10 <- AAPX %>% group_by(Genre)%>%summarise(Avg_Box=round(mean(Box),2))%>%
  arrange(desc(Avg_Box))%>%top_n(10)
datatable(Genre10)

ggplot(Genre10 ,aes(reorder(Genre,Avg_Box),Avg_Box,group=1,fill=Genre))+geom_bar(stat='identity')+
  xlab("Genre")+ylab("Avg_box") +theme(legend.position='None')+coord_flip()+
  theme(axis.ticks = element_blank(),panel.background = element_blank(),
        axis.text.x= element_text(),
        axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5))

ggsave('Genre10.png', width = 20, height = 10, units = "cm")

#Box per Genre,MPAA
MG <- AAPX %>% group_by(MPAA,Genre) %>%summarise(Avg_Box=round(mean(Box),2))%>%ungroup()%>%
  arrange(desc(Avg_Box))
datatable(MG)

MG1 <- AAPX %>% group_by(MPAA,Genre) %>%summarise(Avg_Box=round(mean(Box),2))%>%
  arrange(desc(Avg_Box))

ggplot(MG1,aes(MPAA,Avg_Box,color=MPAA,fill=MPAA))+geom_bar(stat='identity')+
  scale_y_sqrt(limits=c(0,5e+08))+facet_wrap(~Genre)+
  geom_hline(aes(yintercept = 1e+08),color='red',size=0.5)
ggsave('MG1.png', width = 35, height = 35, units = "cm")
# AAPX$Title[which.max(AAPX$Youtube.Views)]

#Box per month
bm <- AAPX %>% group_by(Month) %>%summarise(Avg_Box=round(mean(Box),2))%>%ungroup()%>%
  arrange(desc(Avg_Box))
datatable(bm)

bm1 <- AAPX %>% group_by(Month) %>%summarise(Avg_Box=round(mean(Box),2))%>%
  arrange(desc(Avg_Box))

ggplot(bm1, aes(Month, Avg_Box,fill=Month)) +
  geom_bar(stat='identity') + theme(legend.position = 'top')+
  ggtitle("Avg Box by Month")+ coord_flip()+
  theme(panel.background = element_blank(),
        axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),)
ggsave('bm1.png', width = 20, height = 10, units = "cm")

#Box per month before 2011 vs after 2011
bmb <- AAPX %>% filter(Year<2011 & Box>1e+08)%>%group_by(Month) %>%summarise(Avg_Box=round(mean(Box),2))%>%
  arrange(desc(Avg_Box))

ggplot(bmb, aes(Month, Avg_Box,group=1)) +
  geom_line() + theme(legend.position = 'none')+
  ggtitle("Avg Box by Month")

bma <- AAPX %>% filter(Year>=2011& Box>1e+08)%>%group_by(Month) %>%summarise(Avg_Box=round(mean(Box),2))%>%
  arrange(desc(Avg_Box))

ggplot() + 
  geom_line(data=bmb, aes(x=Month, y=Avg_Box,group=1), color='blue') + 
  geom_line(data=bma, aes(x=Month, y=Avg_Box,group=1), color='red')+
  theme(panel.background=element_blank(),
        axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5))

ggsave('Year2011.png')


#Box per Genre, Month
GenRe <- AAPX %>% group_by(Month,Genre) %>%summarize(Avg_box=round(mean(Box),2))%>%ungroup()%>%
  arrange(desc(Avg_box))
datatable(GenRe)

ggplot(GenRe,aes(Month,Avg_box,color=Month,fill=Month))+geom_bar(stat='identity')+
  scale_y_sqrt(limits=c(0,5e+08))+facet_wrap(~Genre)+geom_hline(aes(yintercept = 1e+08),color='red',size=0.5)+
  theme(axis.text.x=element_text(angle = 90, hjust = 1))
ggsave('GenRe.png', width = 35, height = 35, units = "cm")

#Box per MPAA,Month
Mom <- AAPX %>% group_by(Month,MPAA) %>%summarize(Avg_box=round(mean(Box),2))%>%ungroup()%>%
  arrange(desc(Avg_box))
datatable(Mom)

ggplot(Mom, aes(Month, MPAA, fill = Avg_box)) +
  geom_tile(color = "white") +
  ggtitle("Avg Box by Month and MPAA")
#or 
ggplot(Mom, aes(Month, Avg_box, group = MPAA,color=MPAA)) +
  geom_line() + ggtitle("Avg Box by Month and MPAA")+
  theme(panel.background = element_blank(),
        axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5))

ggsave('Mom.png', width = 20, height = 10, units = "cm")

#Box per distributor
Dis6 <- AAPX %>% group_by(Distrubutor)%>% filter(Distrubutor %in% c('Sony / Columbia',
                     'Warner Bros.','Fox','Universal','Buena Vista','Paramount'))%>%
  summarise(count = n(),avg_box=round(mean(Box),2))%>%ungroup() %>%arrange(desc(avg_box))
datatable(Dis6)

ggplot(Dis6,aes(reorder(Distrubutor,avg_box),avg_box,fill=Distrubutor))+geom_bar(stat='identity')+ 
  theme(legend.position='None')+coord_flip()+ xlab("Big 6 Dis")

#Box per distrubutor in terms of month
Mis6 <- AAPX %>% group_by(Distrubutor,Month)%>% filter(Distrubutor %in% c('Sony / Columbia',
              'Warner Bros.','Fox','Universal','Buena Vista','Paramount'))%>%
  summarise(count = n(),avg_box=round(mean(Box),2))%>%ungroup() %>%arrange(desc(avg_box))
datatable(Mis6)

ggplot(Mis6,aes(reorder(Distrubutor,avg_box),avg_box,fill=Distrubutor))+
  geom_bar(stat='identity')+  facet_wrap(~Month)+ 
  theme(legend.position='None',axis.text.x= element_text(angle = -90, hjust = 0.5))+
  coord_flip()+ xlab("Big 6 Dis")
ggsave('Mis6.png', width = 20, height = 10, units = "cm")

#Box per distrubutor in terms of Year
Yis6 <- AAPX %>% group_by(Distrubutor,Year)%>% filter(Distrubutor %in% c('Sony / Columbia',
                                                                          'Warner Bros.','Fox','Universal','Buena Vista','Paramount'))%>%
  summarise(count = n(),avg_box=round(mean(Box),2))%>%ungroup() %>%arrange(desc(avg_box))
datatable(Yis6)

ggplot(Yis6,aes(Year,avg_box,color=Distrubutor))+
  geom_line(size=0.5)+theme(axis.text.x= element_text(),
                    panel.background = element_blank(),
                    axis.line.x = element_line(color="black", size = 0.5),
                    axis.line.y = element_line(color="black", size = 0.5))+
  xlab("Big 6 Dis")
ggsave('Yis6.png', width = 20, height = 10, units = "cm")

#Youtube vs Box
ggplot(AAPX,aes(Youtube.Views,Box))+geom_point()
#(Year > 2013)
YouB <- AAPX %>%select(Title,Box,Youtube.Views)
ggplot(YouB,aes(Youtube.Views,Box,label=Title))+geom_point()+geom_smooth()+
  geom_text(check_overlap = TRUE)+
  theme(axis.text.x= element_text(),
            panel.background = element_blank(),
            axis.line.x = element_line(color="black", size = 0.5),
            axis.line.y = element_line(color="black", size = 0.5))
ggsave('YouB.png')

YouB1 <- AAPX %>%filter(Year >2012)%>% select(Title,Box,Youtube.Views)
ggplot(YouB1,aes(Youtube.Views,Box,label=Title))+geom_point()+geom_smooth()+
  geom_text(check_overlap = TRUE)+
  theme(axis.text.x= element_text(),
        panel.background = element_blank(),
        axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5))
ggsave('YouB1.png')

#correlation between box & youtube from 2005-2016
correlation <- data.frame()
for (i in c(2005:2016)){  
  AAPA <- AAPL %>% filter(Year>=i)
  x <- AAPA[13]
  y <- AAPA[14]
  corre <- round(cor(x, y),2)
  correlation = rbind(correlation,corre)
}
row.names(correlation) <- c(2005:2016)
colnames(correlation) <- 'r'

ggplot(correlation,aes(rownames(correlation),r,group=1)) +
  geom_line(color='blue')+
  geom_text(data =correlation,aes(x= rownames(correlation),y=r,label=r,vjust=-1,size=3))+
  ggtitle("Corr between Box & Youtube Views")+xlab('Year')+
  theme(panel.background = element_blank(),
        axis.line.x = element_line(color='black',size=0.25),
        axis.line.y = element_line(color='black',size=0.25),
        axis.ticks.x=element_blank(),
        legend.position = 'None')  
#or 
ggplot(correlation,aes(rownames(correlation),r)) +
  geom_bar(stat='identity',fill="#009E73",width=0.25)+
  coord_cartesian(ylim = c(0,1))+
  ggtitle("Corr between Box & Youtube Views")+xlab('Year')+
  geom_text(data =correlation,aes(x= rownames(correlation),y=r,label=r,vjust=-1,size=3))+
  theme(panel.background = element_blank(),
        axis.line.x = element_line(color='black',size=0.25),
        axis.line.y = element_line(color='black',size=0.25),
        axis.ticks.x=element_blank(),
        legend.position = 'None')  
ggsave('corr.png')
# MPAA,Box,Year
MBY <- AAPX %>% group_by(MPAA,Year) %>%summarise(avg_box = round(mean(Box),2)) %>% ungroup()%>%
  arrange(desc(avg_box))
datatable(MBY)

MBY1 <- AAPX %>% group_by(MPAA,Year) %>%summarise(avg_box = round(mean(Box),2)) %>%
  arrange(desc(avg_box))

ggplot(MBY1,aes(Year,avg_box,group=MPAA,color=MPAA))+geom_line()

# Genre ,Box,Year
GBY <- AAPX %>% group_by(Genre,Year) %>%summarise(avg_box = round(mean(Box),2)) %>% ungroup()%>%
  arrange(desc(avg_box))
datatable(GBY)

GBY1 <- AAPX %>% group_by(Genre,Year) %>%summarise(avg_box = round(mean(Box),2)) %>%
  arrange(desc(avg_box))

ggplot(GBY1,aes(Year,avg_box,color=Year,fill=Year))+geom_bar(stat='identity')+
  scale_y_sqrt(limits=c(0,5e+08))+  facet_wrap(~Genre)+geom_hline(aes(yintercept = 1e+08),color='red',size=0.5)
ggsave('GBY1.png', width = 20, height = 20, units = "cm")

#MPAA max box office pk

MPAAmax <- AAPX %>% group_by(MPAA,Year)%>%summarise(Box = max(Box))
MPAAmaxT <- merge(MPAAmax, AAPX, by = "Box")
MPAAmaxT <- MPAAmaxT[,c(1:4)] 
#rename(MPAAmaxT, c("MPAA.x"="MPAA", "Year.x"="Year"))
colnames(MPAAmaxT)[2] <- 'MPAA'
colnames(MPAAmaxT)[3] <- 'Year'
colnames(MPAAmaxT)[1] <- 'maxbox'
ggplot(MPAAmaxT,aes(Year,maxbox,group=MPAA,color=MPAA,label=Title))+geom_line()+
  geom_text(check_overlap = TRUE,data=subset(MPAAmaxT, (MPAA=='R'| MPAA=='PG-13')& 
                                               maxbox >= 2e8L))+
  theme(axis.text.x= element_text(),
        panel.background = element_blank(),
        axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5))
ggsave('MPAAmaxT.png', width = 20, height = 10, units = "cm")

#Year max box office in terms of MPAA pk

MPmax <- AAPX %>% group_by(Year)%>%summarise(Box = max(Box))
MPmaxT <- merge(MPmax, AAPX, by = "Box")
MPmaxT <- MPmaxT[,c(1:3,11)] 
#rename(MPAAmaxT, c("MPAA.x"="MPAA", "Year.x"="Year"))
colnames(MPmaxT)[2] <- 'Year'
colnames(MPmaxT)[1] <- 'maxbox'
ggplot(MPmaxT,aes(Year,maxbox,group=MPAA,fill=MPAA,label=Title))+geom_bar(stat='identity')+
  geom_text(check_overlap = TRUE,data=subset(MPAAmaxT, maxbox >= 3e8L))+  
  theme(axis.text.x= element_text(),
  panel.background = element_blank(),
  axis.line.x = element_line(color="black", size = 0.5),
  axis.line.y = element_line(color="black", size = 0.5))
ggsave('MPmaxT.png', width = 20, height = 10, units = "cm")

#Year max box office in terms of Genre pk

Genremax <- AAPX %>% group_by(Year)%>%summarise(Box = max(Box))
GenremaxT <- merge(Genremax, AAPX, by = "Box")
GenremaxT <- GenremaxT[,c(1:3,9)] 
#rename(MPAAmaxT, c("Genre.x"="Genre", "Year.x"="Year"))
colnames(GenremaxT)[2] <- 'Year'
colnames(GenremaxT)[1] <- 'maxbox'
ggplot(GenremaxT,aes(Year,maxbox,group=Genre,fill=Genre,label=Title))+
  geom_bar(stat='identity')+
  geom_text(check_overlap = TRUE,data=subset(GenremaxT, maxbox >= 4e8L))+
  theme(axis.text.x= element_text(),
        panel.background = element_blank(),
        axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),
        legend.position = 'top')
ggsave('GenremaxT.png', width = 20, height = 10, units = "cm")

# detailed Box per MPAA
library(xts)
library(dygraphs)
backDate <- function(x) as.POSIXct(strptime(x, '%Y-%m-%d'))

xt <- AAPX%>%filter(MPAA=='G')%>%group_by(MPAA,Release.Date)%>%summarise(avg_box=mean(Box))%>% 
  arrange(desc(avg_box))
xt$Release.Date <- backDate(xt$Release.Date)

xt1 <- AAPX%>%filter(MPAA=='PG')%>%group_by(MPAA,Release.Date)%>%summarise(avg_box=mean(Box))%>% 
  arrange(desc(avg_box))
xt1$Release.Date <- backDate(xt1$Release.Date)

xt2 <- AAPX%>%filter(MPAA=='PG-13')%>%group_by(MPAA,Release.Date)%>%summarise(avg_box=mean(Box))%>% 
  arrange(desc(avg_box))
xt2$Release.Date <- backDate(xt2$Release.Date)

xt3 <- AAPX%>%filter(MPAA=='R')%>%group_by(MPAA,Release.Date)%>%summarise(avg_box=mean(Box))%>% 
  arrange(desc(avg_box))
xt3$Release.Date <- backDate(xt3$Release.Date)

xt4 <- AAPX%>%filter(MPAA=='NC-17')%>%group_by(MPAA,Release.Date)%>%summarise(avg_box=mean(Box))%>% 
  arrange(desc(avg_box))
xt4$Release.Date <- backDate(xt4$Release.Date)

dxts <- xts(xt1, order.by=xt1$Release.Date)
dygraph(dxts, main="Box Office per time") %>%
  dySeries("Release.Date", label = "MPAA") %>%
  dyRangeSelector(height = 10)
