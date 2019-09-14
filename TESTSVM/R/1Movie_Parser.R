library(XML)
library(bitops)
library(RCurl)
library(httr)

rm(list=ls(all=TRUE))
Sys.setlocale("LC_ALL", "cht")
yahoourl = "http://www.boxofficemojo.com/yearly/chart/?page="
appendurl_1 = "&view=releasedate&view2=domestic&yr="
appendurl_2 = "&p=.htm"
testurl = ""
testvector = c()
testframe = data.frame()

myHttpheader<- c(
  "User-Agent"="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36",
  "Accept"="text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
  "Connection"="keep-alive",
  "Accept-Charset"="big5,GB2312,utf-8;q=0.7,*;q=0.7",
  "Accept-Encoding"="gzip, deflate, sdch",
  "Accept-Language"="zh-TW,zh;q=0.8,en-US;q=0.6,en;q=0.4",
  "Upgrade-Insecure-Requests"="1",
  "Cache-Control"="max-age=0",
  "Cookie"="__utmt=1; __utma=137419939.1443367072.1468586224.1468720076.1468727749.10; __utmb=137419939.13.10.1468727749; __utmc=137419939; __utmz=137419939.1468600072.4.2.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided)",
  "Host"="www.boxofficemojo.com"
)

for(j in 1980:2016)
{
  for(i in 1:10)
  {
    testurl = paste(yahoourl,i,appendurl_1,j,appendurl_2,sep='')
    testexist = url.exists(testurl)
    print(testurl)
    print(testexist)
    if(testexist)
    {
      html = getURL(testurl, ssl.verifypeer = FALSE, encoding='UTF-8', httpheader = myHttpheader)
      xml = htmlParse(html, encoding='UTF-8')
      path = xpathSApply(xml,'//table//td[2]/b/font[@size="2"]/a/@href',sessionEncoding='UTF-8')
      if(length(path)<1) break
      title = xpathSApply(xml,'//table//td[2]/b/font[@size="2"]/a[@href]',sessionEncoding='UTF-8',xmlValue)
      if(length(title)<1) break
      box = xpathSApply(xml,'//table//td[4]/font[@size="2"]/b',sessionEncoding='UTF-8',xmlValue)
      if(length(box)<1) break
      
      testlen = length(path)
      
      if(length(title)!=testlen || length(box)!=testlen) next
      tempframe = data.frame(title,path,box)
      testframe = rbind(testframe,tempframe)
    }
    else print(paste("assert: URL:",testurl,"not exist!"))
  }
}

names(testframe) = c("Title","Path","Box")

#Post-processing and sorting
testframe$Box = substring(testframe$Box,2)
testframe$Box = gsub(",","",testframe$Box)
testframe$Box = as.numeric(testframe$Box)
testframe = testframe[order(testframe$Box,decreasing=TRUE),]

write.csv(testframe,"testcsv.csv")
