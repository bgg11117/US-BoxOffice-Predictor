#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

test = test_AAPL(AAPL,np=npval)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  output$Genre_rank <- renderPlot({
    h1 = t(data.frame(data.frame(table(test[[2]]$Genre[test[[2]]$Y==input$r_bins]))[2],
                      data.frame(table(test[[2]]$Genre[test[[2]]$svm.pred==input$r_bins]))[2]))
    lbel = t(data.frame(table(test[[2]]$Genre[test[[2]]$Y==input$r_bins]))[1])
    barplot(h1,
            names.arg = lbel,
            col=c('green','red'),
            xlab="Genre", ylab="Number", 
            main=paste("Pred Box(red) vs Box(green)\n","precision: ",test[[1]]),
            legend = c("Real","Predict"),beside=TRUE,
            args.legend = list(title = "Color", x = "topright", cex = .7)
    )
  })
  
  output$Dist_rank <- renderPlot({
    h2 = t(data.frame(data.frame(table(test[[2]]$Distrubutor[test[[2]]$Y==input$r_bins]))[2],
                      data.frame(table(test[[2]]$Distrubutor[test[[2]]$svm.pred==input$r_bins]))[2]))
    lbel = t(data.frame(table(test[[2]]$Distrubutor[test[[2]]$Y==input$r_bins]))[1])
    barplot(h2,
            names.arg = lbel,
            col=c('green','red'),
            xlab="Distrubutor", ylab="Number", 
            main=paste("Pred Box(red) vs Box(green)\n","precision: ",test[[1]]),
            legend = c("Real","Predict"),beside=TRUE,
            args.legend = list(title = "Color", x = "topright", cex = .7)
    )
  })
  
  output$YtB_rank <- renderPlot({
    plot(x=test[[2]]$Youtube.Views,y=test[[2]]$Y,xlab="Youtube",ylab="Rank", main="Youtube vs Box", col='green')
  })
  
  output$Mon <- renderPlot({
    h2 = t(data.frame(data.frame(table(test[[2]]$Release.Date[test[[2]]$Y==input$r_bins]))[2],
                      data.frame(table(test[[2]]$Release.Date[test[[2]]$svm.pred==input$r_bins]))[2]))
    lbel = t(data.frame(table(test[[2]]$Release.Date[test[[2]]$Y==input$r_bins]))[1])
    barplot(h2,
            names.arg = lbel,
            col=c('green','red'),
            xlab="Month", ylab="Number", 
            main=paste("Pred Box(red) vs Box(green)\n","precision: ",test[[1]]),
            legend = c("Real","Predict"),beside=TRUE,
            args.legend = list(title = "Color", x = "topright", cex = .7)
    )
  })
  
  output$LEN <- renderPlot({
    plot(x=test[[2]]$Y,y=test[[2]]$Runtime,xlab="Rank", ylab="Runtime", main="Runtime vs Box", col='green')
  })
  
  output$MPAA_rank <- renderPlot({
    h2 = t(data.frame(data.frame(table(test[[2]]$MPAA[test[[2]]$Y==input$r_bins]))[2],
                      data.frame(table(test[[2]]$MPAA[test[[2]]$svm.pred==input$r_bins]))[2]))
    lbel = t(data.frame(table(test[[2]]$MPAA[test[[2]]$Y==input$r_bins]))[1])
    barplot(h2,
            names.arg = lbel,
            col=c('green','red'),
            xlab="MPAA", ylab="Number", 
            main=paste("Pred Box(red) vs Box(green)\n","precision: ",test[[1]]),
            legend = c("Real","Predict"),beside=TRUE,
            args.legend = list(title = "Color", x = "topright", cex = .7)
    )
  })
  
  output$MPAA3D <- renderPlot({
    #plot3d(test[[2]]$Genre, test[[2]]$MPAA, test[[2]]$Y, main="3D scatterplot", pch=16, highlight.3d = TRUE, type="h", col=c('red','green','blue'))
    scatterplot3d(test[[2]]$Genre, test[[2]]$MPAA, test[[2]]$Y,main="3D scatterplot", pch=16, highlight.3d = TRUE, type="p")
  })
  
  output$make_pred <- renderText({
    #print(typeof(input$Distrubutor))
    pred_data = data.frame(input$Distrubutor,input$Month,input$Genre,input$Rtime,input$MPAA,input$Y_views,0)
    names(pred_data) = names(AAPL)
    AAPL = rbind(AAPL, pred_data)
    pred_result = predict(test[[3]], AAPL[nrow(AAPL),-7])
    print(paste("Possiable rank:",as.numeric(pred_result)))
    #print(paste(input$Y_views,input$Rtime,input$Month,input$MPAA,input$Distrubutor,input$Genre))
  })
  
})
