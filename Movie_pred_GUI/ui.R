#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

# Define UI for application that draws a histogram
shinyUI(navbarPage(
  "Movie box prediction",
  tabPanel("Rank Analaze",
           sidebarLayout(
             sidebarPanel(
               sliderInput("r_bins",
                           "Rank",
                           min = 1,
                           max = 5,
                           value = 2
                           )
             ),
             mainPanel(
               plotOutput("Genre_rank"),
               plotOutput("MPAA_rank"),
               plotOutput("Dist_rank"),
               #plotOutput("YtB_rank"),
               plotOutput("Mon"),
               plotOutput("LEN")
             )
           )
           ),
  tabPanel("MPAA 3D scatterplot",
          mainPanel(
            plotOutput("MPAA3D")
          )
        ),
  tabPanel("Make Predict",
           sidebarLayout(
             sidebarPanel(
              numericInput("Y_views", 
                            label = h3("Youtube views"), 
                            value = 1000),
              numericInput("Rtime", 
                           label = h3("Runtime(min)"), 
                           value = 100),
             selectInput("Month", label = h3("Select Month"), 
                         choices = list("January"="January",
                                        "February"="February",
                                        "March"="March",
                                        "April"="April",
                                        "May"="May",
                                        "June"="June",
                                        "July"="July",
                                        "August"="August",
                                        "September"="September",
                                        "October"="October",
                                        "November"="November",
                                        "December"="December",
                                        "Unknown"=""), selected = 1),
             selectInput("MPAA", label = h3("Select MPAA"),
                         choices = chifun(AAPL$MPAA) 
             ),
             selectInput("Distrubutor", label = h3("Select Distrubutor"),
                         choices = chifun(AAPL$Distrubutor)
             ),
             selectInput("Genre", label = h3("Select Genre"),
                         choices = chifun(AAPL$Genre)
             )
             ),
             mainPanel(
               textOutput("make_pred")
             )
           )
  )
)
)
