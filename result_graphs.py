import matplotlib.pyplot as plt
import numpy as np


data = {}

times =  {'sgd': 0.05437908172607422, 'svrg': 0.24379336833953857, 'saga': 0.06699194908142089, 'sdca': 0.065228271484375}
training_errors =  {'sgd': 0.08888888888888889, 'svrg': 0.08888888888888889, 'saga': 0.08888888888888889, 'sdca': 0.022222222222222223}
testing_errors =  {'sgd': 0.09999999999999999, 'svrg': 0.09999999999999999, 'saga': 0.09999999999999999, 'sdca': 0.09999999999999999}

data[10] = [times, training_errors, testing_errors]


times =  {'sgd': 0.4966533660888672, 'svrg': 2.019904947280884, 'saga': 0.4844032287597656, 'sdca': 0.5720478057861328}
training_errors =  {'sgd': 0.09533333333333334, 'svrg': 0.08222222222222221, 'saga': 0.09111111111111113, 'sdca': 0.07333333333333335}
testing_errors =  {'sgd': 0.076, 'svrg': 0.06000000000000001, 'saga': 0.07, 'sdca': 0.06000000000000001}

data[50] = [times, training_errors, testing_errors]


times =  {'sgd': 6.218096542358398, 'svrg': 30.02920346260071, 'saga': 6.526774644851685, 'sdca': 6.67530255317688}
training_errors =  {'sgd': 0.04195555555555556, 'svrg': 0.036, 'saga': 0.03715555555555556, 'sdca': 0.037333333333333336}
testing_errors =  {'sgd': 0.08080000000000001, 'svrg': 0.096, 'saga': 0.088, 'sdca': 0.088}

data[250] = [times, training_errors, testing_errors]


times =  {'sgd': 83.57937526702881, 'svrg': 412.85660004615784, 'saga': 87.26010131835938, 'sdca': 91.52031540870667}
training_errors =  {'sgd': 0.06766666666666667, 'svrg': 0.07122222222222223, 'saga': 0.07455555555555556, 'sdca': 0.021666666666666667}
testing_errors =  {'sgd': 0.101, 'svrg': 0.102, 'saga': 0.104, 'sdca': 0.081}

data[1000] = [times, training_errors, testing_errors]


methods = ["sgd", "svrg", "saga", "sdca"]
values = [10, 50, 250, 1000]
times = [[data[val][0].get(method, 0) for val in values if method in data[val][0]] for method in methods]
training_errors = [[data[val][1].get(method, 0) for val in values if method in data[val][1]] for method in methods]
testing_errors = [[data[val][2].get(method, 0) for val in values if method in data[val][2]] for method in methods]


plt.yscale("log")
plt.xlabel("Dimension of the samples")
plt.ylabel("Training error")
for i, method in enumerate(methods):
    plt.plot(values[len(values)-len(training_errors[i]):], testing_errors[i], label=method)
plt.legend()
plt.show()
