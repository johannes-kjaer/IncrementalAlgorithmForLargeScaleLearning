import matplotlib.pyplot as plt

data = {}

#m = 100, n = 1000, l = 2, correlation = 0.005, eta = 0.00001, epochs = 50

times =  {'sgd': 0.7047759532928467, 'svrg': 3.251377296447754, 'saga': 0.7749918460845947, 'sdca': 0.6325435161590576}
training_errors =  {'sgd': 0.07088888888888889, 'svrg': 0.06888888888888889, 'saga': 0.06888888888888889, 'sdca': 0.06888888888888889}
testing_errors =  {'sgd': 0.118, 'svrg': 0.11000000000000001, 'saga': 0.12, 'sdca': 0.12}
data[0.005] = [times, training_errors, testing_errors]


#m = 100, n = 1000, l = 2, correlation = 0.01, eta = 0.00001, epochs = 50

times =  {'sgd': 0.7079001426696777, 'svrg': 3.2809158325195313, 'saga': 0.771441125869751, 'sdca': 0.632981252670288}
training_errors =  {'sgd': 0.07933333333333333, 'svrg': 0.07888888888888888, 'saga': 0.0811111111111111, 'sdca': 0.07777777777777778}
testing_errors =  {'sgd': 0.06, 'svrg': 0.06, 'saga': 0.06, 'sdca': 0.06}
data[0.01] = [times, training_errors, testing_errors]


#m = 100, n = 1000, l = 2, correlation = 0.03, eta = 0.00001, epochs = 50

times =  {'sgd': 0.7089083671569825, 'svrg': 3.2637966156005858, 'saga': 0.7753928184509278, 'sdca': 0.6455083847045898}
training_errors =  {'sgd': 0.029777777777777775, 'svrg': 0.028888888888888888, 'saga': 0.03, 'sdca': 0.03}
testing_errors =  {'sgd': 0.042, 'svrg': 0.05, 'saga': 0.05, 'sdca': 0.05}
data[0.03] = [times, training_errors, testing_errors]


#m = 100, n = 1000, l = 2, correlation = 0.05, eta = 0.00001, epochs = 50

times =  {'sgd': 0.7046433448791504, 'svrg': 3.2982466220855713, 'saga': 0.7966178894042969, 'sdca': 0.6561943054199219}
training_errors =  {'sgd': 0.027555555555555555, 'svrg': 0.026666666666666665, 'saga': 0.026666666666666665, 'sdca': 0.026666666666666665}
testing_errors =  {'sgd': 0.01, 'svrg': 0.01, 'saga': 0.01, 'sdca': 0.01}
data[0.05] = [times, training_errors, testing_errors]


#m = 100, n = 1000, l = 2, correlation = 0.1, eta = 0.00001, epochs = 50

times =  {'sgd': 0.7203641891479492, 'svrg': 3.241556453704834, 'saga': 0.776317024230957, 'sdca': 0.6553245067596436}
training_errors =  {'sgd': 0.01422222222222222, 'svrg': 0.014444444444444444, 'saga': 0.014444444444444444, 'sdca': 0.014444444444444444}
testing_errors =  {'sgd': 0.02, 'svrg': 0.02, 'saga': 0.02, 'sdca': 0.02}
data[0.1] = [times, training_errors, testing_errors]


#m = 100, n = 1000, l = 2, correlation = 0.5, eta = 0.00001, epochs = 50

times =  {'sgd': 0.7106057643890381, 'svrg': 3.2674859523773194, 'saga': 0.7750696659088134, 'sdca': 0.704461669921875}
training_errors =  {'sgd': 0.0044444444444444444, 'svrg': 0.0044444444444444444, 'saga': 0.0044444444444444444, 'sdca': 0.0044444444444444444}
testing_errors =  {'sgd': 0.0, 'svrg': 0.0, 'saga': 0.0, 'sdca': 0.0}
data[0.5] = [times, training_errors, testing_errors]


#m = 100, n = 1000, l = 2, correlation = 0.9, eta = 0.00001, epochs = 50

times =  {'sgd': 0.7156815528869629, 'svrg': 3.278140354156494, 'saga': 0.771565580368042, 'sdca': 0.7508546352386475}
training_errors =  {'sgd': 0.0, 'svrg': 0.0, 'saga': 0.0, 'sdca': 0.0}
testing_errors =  {'sgd': 0.0, 'svrg': 0.0, 'saga': 0.0, 'sdca': 0.0}
data[0.9] = [times, training_errors, testing_errors]


methods = ["sgd", "svrg", "saga", "sdca"]
values = [0.005, 0.01, 0.03, 0.05, 0.1, 0.5, 0.9]
times = [[data[val][0].get(method, 0) for val in values if method in data[val][0]] for method in methods]
training_errors = [[data[val][1].get(method, 0) for val in values if method in data[val][1]] for method in methods]
testing_errors = [[data[val][2].get(method, 0) for val in values if method in data[val][2]] for method in methods]

plt.xscale('log')
plt.xlabel("Correlation parameter")
plt.ylabel("Testing errors")
for i, method in enumerate(methods):
    plt.plot(values[len(values)-len(testing_errors[i]):], testing_errors[i], label=method)
plt.legend()
plt.show()