# MNIST benchmark

benchmarks of different machine learning libraries using the MNIST dataset
- (tensorflow) DNN : 97.81%
- (tensorflow) CNN : coming up
- PyTorch          : coming up

### Prerequisites

tensorflow:
```
$ pip3 install tensorflow
```
note: has to be Python3.7.6 or older, tensorflow is not updated for Python3.8 yet

### Use

```
$ python3 benchmark.py
```

### predictions.py

played around with plotting
- incorrect classifications
  - network guess (red) and certainty (bar graph)
  - actual classification (blue)
- unconfident predictions (< 65% certain)


## Authors

- **simon** - *buy my merch* - [mightbesimon](https://github.com/mightbesimon)

Shoutout to tensorflow.org tutorials\
Shoutout to Google for creating tensorflow

## Acknowledgments

- I'm just following tensorflow.org tutorials
