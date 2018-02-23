# In professional sports, one of the biggest questions that management has to continually evaluate is
# whether or not the players they are investing in are providing tangible, measureable results.  This is 
# a fundamental concept that plays an integral role in trade considerations, contract negotations, and 
# many other aspects of team management.  While it's difficult to accurately determine a player's value to 
# a team over the course of just one season (injuries, personnel changes, individual roles on the team, etc.
# vary from year to year), it is nevertheless useful for teams to gather performance metrics and compare 
# their players' performance and salaries to those of other players around the league.  

# In this project, I'll model NHL player salaries as a function of numerous statistical categories 
# compiled over the course of the 2016-2017 regular season and use that information to examine just how 
# much value each team got from it's collective playerbase.  

# Loading required packages

require(keras)
require(caret)
require(xgboost)
require(plyr)
require(tidyverse)
require(ggplot2)
require(lubridate)
require(scales)
require(rvest)
require(onehot)

# Loading in training and testing data and then temporarily combining them into one data.frame

set.seed(314)
train <- read.csv("C:\\Users\\Bryan\\Google Drive\\Kaggle\\Hockey\\train.csv", stringsAsFactors = FALSE)
test.x <- read.csv("C:\\Users\\Bryan\\Google Drive\\Kaggle\\Hockey\\test.csv", stringsAsFactors = FALSE)
test.y <- read.csv("C:\\Users\\Bryan\\Google Drive\\Kaggle\\Hockey\\test_salaries.csv", stringsAsFactors = FALSE)
test <- cbind(test.y, test.x)
rm(test.x); rm(test.y)
all.players <- rbind(train, test)
rm(test); rm(train)

# There is a lot of potentially significant information to gather from a player's date of birth.  Extracting
# age and month of birth from DOB.

all.players$Born <- ymd(all.players$Born)
all.players <- all.players %>% mutate(exactage = time_length(as.duration(interval(all.players$Born, Sys.Date())), "year"),
                                      monthbirth = as.factor(month(all.players$Born)),
                                      yearbirth = as.numeric(year(all.players$Born))) %>% 
                               mutate(floorage = floor(exactage)) %>%
                               select(-City)

# The state in which each player is from isn't applicable to every country.  Adding "INT" factor to those
# entries.

all.players$Pr.St[all.players$Pr.St == ""] <- "INT"
all.players$Pr.St <- as.factor(all.players$Pr.St)

# Some players are from countries from which where there are not many NHL players. We'll combine those countries
# under one factor heading.

low.pop <- c("AUT", "DEU", "DNK", "EST", "FRA", "GBR", "HRV", "ITA", "NOR", "SVN", "LVA")
all.players <- all.players %>% mutate(Cntry = ifelse(Cntry %in% low.pop, "INT", Cntry),
                                      Nat = ifelse(Nat %in% low.pop, "INT", Cntry))

# Converting other variables to factors

all.players$Cntry <- as.factor(all.players$Cntry)
all.players$Nat <- as.factor(all.players$Nat)

# Creating additional factor levels to account for players who were undrafted

all.players$DftYr[is.na(all.players$DftYr)] <- 0
all.players$DftRd[is.na(all.players$DftRd)] <- 0
all.players$Ovrl[is.na(all.players$Ovrl)] <- max(all.players$Ovrl, na.rm = TRUE) + 1
temp.draft <- data.frame(year(all.players$Born), all.players$DftYr)
names(temp.draft) <- c("born", "draft")
temp.draft <- temp.draft[temp.draft$draft != 0,]
temp.draft <- apply(temp.draft, 2, as.numeric)
temp.draft <- round(aggregate(draft ~ born, data = temp.draft, mean))
mean.draft.age <- round(mean(temp.draft$draft - temp.draft$born)) + 1
all.players <- all.players %>% mutate(leagueyears = ifelse(DftYr == 0, (as.numeric(yearbirth) + mean.draft.age), (2017 - as.numeric(DftYr))),
                                      DftYr = as.numeric(DftYr),
                                      DftRd = as.numeric(DftRd))
rm(temp.draft)
all.players <- select(all.players, -Born)

# One hotting factor variable

all.players <- all.players %>% mutate(Hand = ifelse(Hand == "L", 1, 0))

# Some players played multiple positions, so additional features have to be added to account for that

all.players <- all.players %>% mutate(numpos = str_count(Position, "/") + 1) %>%
                               separate(col = Position, 
                                        into = c("primpos", "secpos", "tertpos"), 
                                        sep = "/", 
                                        fill = "right")
all.players <- all.players %>% mutate(lwpos = ifelse(all.players$primpos == "LW" | 
                                                     all.players$secpos == "LW" | 
                                                     all.players$tertpos == "LW", 1, 0),
                                      rwpos = ifelse(all.players$primpos == "RW" | 
                                                     all.players$secpos == "RW" | 
                                                     all.players$tertpos == "RW", 1, 0),
                                      cpos = ifelse(all.players$primpos == "C" | 
                                                    all.players$secpos == "C" | 
                                                    all.players$tertpos == "C", 1, 0),
                                      dpos = ifelse(all.players$primpos == "D" | 
                                                    all.players$secpos == "D" | 
                                                    all.players$tertpos == "D", 1, 0))
all.players$lwpos[is.na(all.players$lwpos)] <- 0
all.players$rwpos[is.na(all.players$rwpos)] <- 0
all.players$cpos[is.na(all.players$cpos)] <- 0
all.players$dpos[is.na(all.players$dpos)] <- 0
all.players <- within(all.players, rm(primpos, secpos, tertpos))

# The amount of money that teams can afford to spend on players is influenced by global and local factors.
# Macroeconomic international considerations such as exchange rates, taxes, and so forth have historical
# relevance to salaries large and small.  Most NHL teams are based in the United States, but some are 
# located in Canada; this is something that should be noted.

canadian.teams <- c("TOR", "VAN", "CGY", "WPG", "OTT", "MTL", "EDM")

# Additionally, because of in-season trades, some players played for multiple teams over the course 
# of the season.  

all.players <- all.players %>% mutate(numteam = str_count(Team, "/") +1) %>%
                               separate(col = Team,
                                        into = c("team1", "team2"),
                                        sep = "/",
                                        extra = "drop",
                                        fill = "right") %>%
                               mutate(team2 = ifelse(is.na(team2), team1, team2)) %>%
                               mutate(team1can = ifelse(team1 %in% canadian.teams, 1, 0),
                                      team2can = ifelse(team2 %in% canadian.teams, 1, 0),
                                      team1 = as.factor(team1),
                                      team2 = as.factor(team2))

# Combining name fields 

all.players <- all.players %>% mutate(fullname = paste(First.Name, Last.Name, sep = " ")) %>%
                               select(-First.Name, -Last.Name)

# One intangible factor in determining a player's value to a team is in his ability to lead.  Team captains
# and alternates are generally chosen because those players are the ones that other's look to for 
# leadership.  The original data did not contain this information, so it had to be scraped from another site.

caps <- read_html("https://www.sporcle.com/games/Flyersfan16/2016-17-nhl-captain-and-alternate-captain-nationality")
caps <- rbind(caps %>% html_nodes("table") %>% .[[2]] %>% html_table(),
              caps %>% html_nodes("table") %>% .[[3]] %>% html_table(),
              caps %>% html_nodes("table") %>% .[[4]] %>% html_table())
caps <- select(caps, -Country)
names(caps) <- c("team", "name")
caps <- separate(caps, team, into = c("teamname", "type"), sep = ";")
caps$type <- trimws(caps$type)
all.players <- merge(all.players, caps, by.x = "fullname", by.y = "name", all.x = TRUE, all.y = FALSE)
all.players <- select(all.players, -teamname)
all.players$type <- as.numeric(gsub(".*a.*", 1, all.players$type))
all.players$type[is.na(all.players$type)] <- 0
rm(caps)

# Converting several fields to 'numerical' data

ints <- which(sapply(1:ncol(all.players), function(x) is.integer(all.players[,x])))
for (i in ints) {
      all.players[,i] <- as.numeric(all.players[,i])
}

# Separating player names from data set

p.names <- all.players$fullname
all.players <- select(all.players, -fullname, -Pr.St)

# One-hotting factor variables

factor.vars <- which(sapply(1:ncol(all.players), function(x) class(all.players[,x])) == "factor")
oh <- onehot(all.players[,factor.vars], max_levels = 32)
oh.df <- as.data.frame(predict(oh, all.players))
all.players <- as.data.frame(cbind(all.players, oh.df))
all.players <- all.players[,-factor.vars]
rm(oh.df)

# Imputing NA data with median values for each corresponding feature

nas <- which(colSums(is.na(all.players)) > 0)
for (i in nas) {
        all.players[is.na(all.players[,i]),i] <- as.numeric(median(all.players[,i], na.rm = TRUE))
}

# Removing highly correlated fields

Salary <- all.players$Salary
correl <- cor(all.players)
all.players <- all.players[,findCorrelation(correl, cutoff = 0.95)]
all.players <- cbind.data.frame(all.players, Salary)

# Splitting data back into training, validation and test sets

intrain <- createDataPartition(all.players$Salary, p = 0.7, list = FALSE)
test.set <- all.players[-intrain,]
train.set <- all.players[intrain,]
train.x <- select(train.set, -Salary)
train.y <- select(train.set, Salary)
test.x <- select(test.set, -Salary)
test.y <- select(test.set, Salary)

# 
# 
# nas <- which(colSums(is.na(train.x)) > 0)
# 
# for (i in nas) {
#       train.x[is.na(train.x[,i]),i] <- as.numeric(median(train.x[,i], na.rm = TRUE))
# }
# 
# nas <- which(colSums(is.na(test.x)) > 0)
# 
# for (i in nas) {
#       test.x[is.na(test.x[,i]),i] <- as.numeric(median(test.x[,i], na.rm = TRUE))
# }
# 
# nas <- which(colSums(is.na(val.x)) > 0)
# 
# for (i in nas) {
#         val.x[is.na(val.x[,i]),i] <- as.numeric(median(val.x[,i], na.rm = TRUE))
# }

# Identifying and removing highly correlated features
# 
# correl <- cor(train.x)
# train.x <- train.x[,-findCorrelation(correl, cutoff = 0.95)]
# test.x <- test.x[,-findCorrelation(correl, cutoff = 0.95)]
# val.x <- val.x[,-findCorrelation(correl, cutoff = 0.95)]

# Scaling the data 

train.x <- scale(train.x)
test.x <- scale(test.x)

# Recombining datasets for use in random forest and boosted tree training algorithms

tr <- cbind(train.x, train.y)
te <- cbind(test.x, test.y)

# Creating initial random forest model

tg <- expand.grid(mtry = c(2,9,40,75))

rf <- train(Salary ~.,
            data = tr,
            tuneGrid = tg,
            method = "rf",
            importance = TRUE,
            ntrees = 1000,
            preProcess = NULL,
            trControl = trainControl(method = "cv"))

# Computing root mean square error for random forest model

rf.predict <- predict(rf, newdata = te)
RMSE(rf.predict, te$Salary)  

# Creating initial boosted tree model

tg <- expand.grid(nrounds = 500, 
                  max_depth = 150,
                  eta = c(0.005, 0.010, 0.015),
                  gamma = 0,
                  colsample_bytree = 0.7,
                  min_child_weight = 1,
                  subsample = 0.1)

xt <- train(Salary ~.,
            data = tr,
            method = "xgbTree",
            tuneGrid = tg,
            importance = TRUE,
            preProcess = NULL,
            trControl = trainControl(method = "cv"))

# Computing root mean square error for boosted tree model

xt.predict <- predict(xt, te)
RMSE(xt.predict, te$Salary)

# Converting the data into matrix format so that it is usable by Keras neural network model

k.train.x <- as.matrix(train.x)
k.train.y <- as.matrix(train.y)
k.test.y <- as.matrix(test.y)
k.test.x <- as.matrix(test.x)

k.model <- keras_model_sequential() %>%
      layer_dense(units = 1024, activation = "relu", input_shape = dim(k.train.x)[[2]], kernel_regularizer = regularizer_l1(0.001)) %>%
      layer_dense(units = 1024, activation = "relu", kernel_regularizer = regularizer_l1(0.001)) %>%
      layer_dense(units = 1)

k.model %>% compile(optimizer = "rmsprop",
                    loss = "mse",
                    metric = c("mae"))

k.history <- k.model %>% fit(k.train.x, 
                            k.train.y, 
                            epochs = 700,
                            batch_size = 64, 
                            verbose = 0,
                            validation_split = 0.3)
k.history
plot(k.history)
k.predict <- predict(k.model, k.test.x)
RMSE(k.predict, k.test.y)


# For the 2016-2017 NHL Season, the league minimum salary was $575,000, so any model predictions that
# indicate a salary below that amount should be altered.

league.min <- min(te$Salary)

# Creating a data.frame that compares all 3 models and determines which one is closest for each player

closer.model <- function(diffxt, diffrf, diffk) {
      m <- "k"
      if (diffxt < diffrf & diffxt < diffk) {m <- "xt"}
      if (diffrf < diffxt & diffrf < diffk) {m <- "rf"}
      return(m)
}

ens <- data.frame(xt.predict, rf.predict, k.predict, te$Salary)
ens <- ens %>% mutate(xt.predict = ifelse(xt.predict < league.min, league.min, xt.predict),
                      rf.predict = ifelse(rf.predict < league.min, league.min, rf.predict),
                      k.predict = ifelse(k.predict < league.min, league.min, k.predict)) %>%
               mutate(xt.diff = abs(te.Salary - xt.predict),
                      rf.diff = abs(te.Salary - rf.predict),
                      k.diff = abs(te.Salary - k.predict)) %>%
ens <- ens %>% mutate(closer.mod = sapply(1:nrow(ens), function(x) closer.model(ens$xt.diff[x], ens$rf.diff[x], ens$k.diff[x])))

# A quick plot comparing the three models. I was hoping to see an easily recognizable trend that would 
# indicate that one model performed better on certain salary ranges compared to the others. The xgbTree 
# model performed better at the very low end of the salary range whereas the random forest model did better 
# just above that.

p <- ggplot(ens, aes(te.Salary)) + geom_histogram(bins = 38) 
p <- p + facet_wrap(~closer.mod) 
p <- p + scale_x_discrete(limits = c(seq(5e5,1e7, 1e6)), labels = comma)
p <- p + theme(axis.text.x = element_text(angle = 45, vjust = 0.8, hjust = 1))
plot(p)

# Another plot that compares the two predictive models.  The red line indicates the hypothetical points where
# the two models agree with each other.  Point size is relative to actual player salaries, so theoretically, the
# smaller points should appear to the left side of each graph and the larger points should appear further
# to the left.  As you can see, that's certainly not always the case.  

facet.names <- c('k' = "Data where neural network model provided better prediction",
                 'rf' = "Data where Random Forest model provided better prediction", 
                 'xt' = "Data where XGBTree model provided better prediction")

p <- ggplot(ens, aes(rf.predict, xt.predict)) + geom_point(color = log(ens$te.Salary/1e5), size = log(ens$te.Salary/5e5))
p <- p + facet_wrap(~closer.mod, labeller = as_labeller(facet.names))
p <- p + scale_x_discrete(limits = c(seq(1e6, 7e6, 1e6)), labels = comma)
p <- p + scale_y_discrete(limits = c(seq(1e6, 7e6, 1e6)), labels = comma)
p <- p + theme(axis.text.x = element_text(angle = 45, vjust = 0.8, hjust = 1))
p <- p + geom_abline(intercept = 0, slope = 1, color = "red")
p <- p + xlab("Random Forest Model Prediction") + ylab("Boosted Tree Model Prediction") 
p <- p + ggtitle("Plotting Random Forest vs Boosted Tree Predictions")
p <- p + theme(plot.title = element_text(hjust = 0.5))
p <- p + theme(panel.margin = unit(2, "lines"))
plot(p)

# Using the models to re-predict the entire dataset. There will obviously be a fair bit of overfitting here, but 
# without data from other years, there's no other way to compare a player's performance over the course of a 
# season.

all.xt.predict <- predict(xt, data.frame(rbind(te,tr)))
all.rf.predict <- predict(rf, data.frame(rbind(te,tr)))
all.k.predict <- predict(k.model, data.frame(rbind(te,tr)))

# Aggregating total team salaries and predicted salaries based on performance metrics.  For players who were 
# traded during the course of the season, their projected and actual salaries were halved and those values
# were applied to both teams they played for.

predict.roster <- data.frame(cbind(all.players[,c("team1", "team2", "numteam", "Salary")], all.xt.predict, all.rf.predict))
predict.roster <- predict.roster %>% mutate(all.xt.predict = ifelse(numteam > 1, all.xt.predict/2, all.xt.predict),
                                            all.rf.predict = ifelse(numteam > 1, all.rf.predict/2, all.rf.predict),
                                            Salary = ifelse(numteam > 1, Salary/2, Salary)) %>%
                                     select(-numteam)
bp.roster <- select(predict.roster, -team2)
ep.roster <- select(predict.roster, -team1)
names(bp.roster) <- names(ep.roster) <- c("team", "Salary", "all.xt.predict", "all.rf.predict")
predict.roster <- unique(rbind(bp.roster, ep.roster))
predict.roster <- aggregate(.~team, data = predict.roster, FUN = sum)

# Ultimately, the goal of each team is to make the playoffs.  Creating a table that shows actual team 
# salaries and projected salaries and labels teams that made the playoffs.

playoff.teams <- c("MTL", "NYR", "OTT", "BOS", "WSH", "TOR", "CBJ", "PIT", "CHI", "NSH", "MIN", "STL", "ANA", "CGY", "EDM", "S.J")

predict.roster <- predict.roster %>% mutate(xt.diff = all.xt.predict - Salary,
                                            rf.diff = all.rf.predict - Salary,
                                            playoffs = as.factor(ifelse(team %in% playoff.teams, "1", "0")))
