* Exercise 2 - Testing the assumptions of the regression error term
* Data: Philippines, 1980-2019 (World Bank, World Development Indicators)
*   cons     = household final consumption expenditure, billion pesos (constant prices)
*   gdp      = gross domestic product, billion pesos (constant prices)
*   lendrate = bank lending interest rate, percent per year
import delimited "exercise2_data.csv", clear
tsset year

* Item 3: fit the regression model
reg cons gdp lendrate

* Item 4: generate the error term and check A.3, E(e)=0
predict error, residual
mean error

* Item 5: normality of the error term (A.6)
sktest error

* Item 6: histogram of the error term with normal curve
histogram error, normal

* Item 7: exogeneity of X1 and X2 with respect to the error term (A.7)
pwcorr error gdp lendrate, sig

* Item 8: homoscedasticity (A.4): plot error square against yhat
predict yhat
generate errorsq = error^2
scatter errorsq yhat

* Item 9: independence of the error terms (A.5)
generate errorlag = error[_n-1]
pwcorr error errorlag, sig

* Item 10: independence of the regressors (A.8)
pwcorr gdp lendrate, sig
