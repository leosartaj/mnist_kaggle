## Digit Recognizer Competition on kaggle

https://www.kaggle.com/c/digit-recognizer

### Best Score

Currently output10.csv is the best submission. (0.98314)

It can be obtained by converting image to black-white scale followed by PCA and scikit learn's SVC.
The code is in notebook named Attempt 4, at the end.


### Untried Ideas:

1. Try thresholding pixel values. Say every value less than 10 is 0, and others are 1. Try on different thresholding levels
2. Try CNN.
