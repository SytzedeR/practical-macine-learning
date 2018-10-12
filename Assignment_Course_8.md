### Coursera Data Science Specialisation

### Course \#8

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, we will use
data from accelerometers on the belt, forearm, arm, and dumbell of 6
participants. They were asked to perform barbell lifts correctly and
incorrectly in 5 different ways. More information is available from the
website here:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>
(see the section on the Weight Lifting Exercise Dataset). We will build
a model to predict the way the barbell is lift.

First, we library the needed packages.

    library(ggplot2); library(caret); library(dplyr); library(rattle);
    library(parallel); library(doParallel); options(digits = 4)

Then we download the data.

    training <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),
                         na.strings = c("NA", "", "#DIV/0!"))
    validation <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"),
                           na.strings = c("NA", "", "#DIV/0!"))

And have a look at the training dataset. This is a large dataset, so
results are not shown in the markdown file.

    str(training)

It appears that the first columns have metadata that are not needed for
analysis. There are also many NA's. We remove these columns from the
dataset to get tidy data.

    training <- select(training, 
          -X, -raw_timestamp_part_1,-raw_timestamp_part_2,
          -cvtd_timestamp, -new_window, -num_window)
    training <- training[,colSums(is.na(training))==0]

Then we split the training dataset in two, one for training and one for
testing.

    set.seed(12345)
    inTrain <- createDataPartition(y=training$classe,
                                   p=0.7, list = F)
    testing <- training[-inTrain,]
    training <- training[inTrain,]
    dim(training); dim(testing)

    ## [1] 13737    54

    ## [1] 5885   54

We will build three different models and test them for accuracy. First
we build a decision tree model.

    modFitTrees <- train(data = training, classe ~ ., method = "rpart")

Let's make a figure of the tree to see what it looks like.

    fancyRpartPlot(modFitTrees$finalModel)

![](Assignment_Course_8_files/figure-markdown_strict/unnamed-chunk-7-1.png)

Now we'll test the model with the testing set.

    predTrees <- predict(modFitTrees, newdata = testing)
    cmTrees <- confusionMatrix(predTrees, testing$classe)
    cmTrees$table

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1494  470  467  438  141
    ##          B   21  380   29  184  147
    ##          C  128  289  530  342  277
    ##          D    0    0    0    0    0
    ##          E   31    0    0    0  517

    cmTrees$overall[1]

    ## Accuracy 
    ##   0.4963

The accuracy of this model is clearly very low, 0.4963. The out of
sample error is 0.5037. We need a better model than this. Now let's try
a Bagging model and test in on the testing data set.

    modFitBagging <- train(data = training, classe ~ ., method = "treebag")
    predBagging <- predict(modFitBagging, newdata = testing)
    cmBagging <- confusionMatrix(predBagging, testing$classe)
    cmBagging$table

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1667   17    2    3    1
    ##          B    7 1111    5    1    4
    ##          C    0    8 1013   17    2
    ##          D    0    0    6  941    6
    ##          E    0    3    0    2 1069

    cmBagging$overall[1]

    ## Accuracy 
    ##   0.9857

This model is already much better, with an accuracy of 0.9857. This
means that for this model the out of sample error is 0.0143. The final
model we try is a random forest model. Building this model can take
quite a while, so we use extra cores as described here:
"<https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md>".
We will also create a training control to further improve speed. Then
we'll test the model with the testing data set.

    cluster <- makeCluster(detectCores()-1)
    registerDoParallel(cluster)
    tc <- trainControl(method = "cv", number = 5, allowParallel = T)
    modFitRF <- train(data = training, classe ~ ., method = "rf", trControl = tc)
    stopCluster(cluster); registerDoSEQ()
    predRF <- predict(modFitRF, newdata = testing)
    cmRF <- confusionMatrix(predRF, testing$classe)
    cmRF$table

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672   13    0    0    0
    ##          B    2 1121   18    0    0
    ##          C    0    5 1005   25    0
    ##          D    0    0    3  939    3
    ##          E    0    0    0    0 1079

    cmRF$overall[1]

    ## Accuracy 
    ##   0.9883

The accuracy of the random forest model is 0.9883. The out of sample
error is 0.0117. This makes it the best model of the three. We will use
this model to predict the way the barbell was lift for the 20 cases in
the validation data set.

    predVal <- predict(modFitRF, newdata = validation)
    print(predVal)

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
