rm(list=ls(all=TRUE))
library(XML)
library(bitops)
library(RCurl)
library(NLP)
library(httr)
library(dplyr)
library(tidyr)
library(devtools)
library(chron)
library(lubridate)

Sys.setlocale(category = "LC_TIME", locale = "English_United States.1252")

alldata = read.csv('testcsv.csv')
orgURL = 'http://www.boxofficemojo.com'
fulldata = data.frame()


for( i in 1:length(alldata$X))
{
  yahooURL <- paste(orgURL, alldata$Path[i], sep='')
  URLExist = url.exists(yahooURL)
  if(URLExist)
  {
    html = getURL(yahooURL, ssl.verifypeer = FALSE)
    xml = htmlParse(html, encoding='UTF-8')
    text = xpathSApply(xml, '//tr[@bgcolor="#ffffff"]/td[@valign="top"]/b',xmlValue)
    if(length(text)!=6) next
    testframe = as.data.frame(t(text))
    names(testframe) = c("Distrubutor","Release Date","Genre","Runtime","MPAA","Budget")
    testframe = cbind(alldata[i,-1],testframe)
    fulldata = rbind(fulldata, testframe)
  }
}


#Post-processing
fulldata <- fulldata[!duplicated(fulldata),]
fulldata <- fulldata %>% arrange(desc(Box))

fulldata$Runtime = gsub(" hrs. ",":",fulldata$Runtime)
fulldata$Runtime = gsub(" min.",":00",fulldata$Runtime)
fulldata$Runtime = times(fulldata$Runtime)
fulldata$Runtime = hour(hms(fulldata$Runtime))*60 + minute(hms(fulldata$Runtime))

fulldata$`Release Date`<- as.Date(fulldata$`Release Date`,format="%B %d,%Y")

fulldata$Year <- format(fulldata$`Release Date`,format='%Y')
fulldata$Month <- format(fulldata$`Release Date`,format='%b')
fulldata$Day <- format(fulldata$`Release Date`,format='%d')

fulldata <- fulldata[,c(1:2,5,10:12,4,6:9,3)]
fulldata$Budget = substring(fulldata$Budget,2)
fulldata$Budget = gsub(" million","",fulldata$Budget)
fulldata$Budget = as.numeric(fulldata$Budget)

write.csv(fulldata,"Fulllist.csv")

