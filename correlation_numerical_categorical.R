# http://stats.stackexchange.com/questions/108007/correlations-with-categorical-variables
N <- 100
num1 <- runif(N)
cat1 <- as.factor(sample(2, N, replace=T)) 
levels(cat1)[1] <- "a"
levels(cat1)[2] <- "b"

num2 <- runif(N)

cat2 <- as.factor(sample(2, N, replace=T))
levels(cat2)[1] <- "x"
levels(cat2)[2] <- "y"

##
cor(num1, num2)

##
df <- data.frame(num=num1, cat=cat1)
aov_results <- aov(num~cat, data=df)
summary(aov_results)
coefficients(aov_results)

##
chisq.test(cat1, cat2)

##
require('psych')
mtr <- cbind(df$num[df$cat=='a'], df$num[df$cat=='b'])
ICC(mtr)

##
require('vcd')
tb <- table(cat1, cat2)
assocstats(tb)
