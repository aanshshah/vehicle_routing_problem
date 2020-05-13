#from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.decomposition import PCA
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.cluster import KMeans, DBSCAN
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import LeaveOneOut, cross_val_score 
#from sklearn.preprocessing import StandardScaler
#import numpy as np
#import pandas as pd
import os
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
#from mpl_toolkits.mplot3d import Axes3D
#import warnings
#from joblib import dump, load
#import json
#import pickle

class StrategyInfo:
    def __init__(self, name, logname):
        self.logname = logname
        self.performance = {} #instance -> distance
        self.name = name
        self.time = {} #instance -> time
        self.read_log()

    @staticmethod
    def filter_name(instance_name):
        instance_name = instance_name.split('/')
        if len(instance_name) > 1:
            instance_name = instance_name[-1]
        else:
            instance_name = instance_name[0]
        return instance_name

    def read_log(self):
        with open(self.logname, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split()
            instance_name = self.filter_name(line[1])
            distance = float(line[5])
            time = float(line[3])
            self.time[instance_name] = time
            self.performance[instance_name] = distance

def get_instance_details(instance_name):
    with open('input/{0}'.format(instance_name), 'r') as f:
        lines = f.readlines()
    numCustomers, numVehicles, vehicleCapacity = [int(x) for x in lines[0].split()]
    return numCustomers, numVehicles, vehicleCapacity

def compare_performance(instance_one, instance_two):
    best_performances = {}
    assert len(instance_one.performance) == len(instance_two.performance)
    instances = list(instance_one.performance.keys())
    for instance in instances:
        dist_one = instance_one.performance[instance]
        dist_two = instance_two.performance[instance]
        if dist_one < dist_two:
            best_performances[instance] = (instance_one.name, dist_one)
        else:
            best_performances[instance] = (instance_two.name, dist_two)
    return best_performances

def generate_inputs(best_performance):
    #0 -> simulated_annealing
    #1 -> combined_hill
    X = []
    labels = []
    instance_names = []
    for instance_name, strat_info in best_performance.items():
        strategy_name, distance = strat_info
        label = 0 if strategy_name == 'simulated_annealing_2opt' else 1
        X.append(get_instance_details(instance_name))
        labels.append(label)
        instance_names.append(instance_name)
    return X, labels, instance_names

def pca(X): #62.5%
    model = PCA().fit(X)
    transformed = model.transform(X)
    return transformed

def KNN(X, y, instance_names, k=8): #68.5
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X, y)
    y_pred = neigh.predict(X)
    score, incorrect_names = accuracy(y_pred, y, instance_names)
    score = neigh.score(X, y)
    return neigh, score, incorrect_names
    # print(correct, incorrect_names)

def randomforest(X, y, instance_names, strategy_names=None):
    clf = RandomForestClassifier(max_depth=2, n_estimators=100, random_state=0).fit(X,y)
    score = clf.score(X,y) 
    pred_label = clf.predict(X)
    correct, incorrect_names = accuracy(pred_label, y, instance_names, strategy_names)
    return clf, correct, incorrect_names

def kmeans(X, y, instance_names, k=4): #62.5%
    model = KMeans(n_clusters=k, random_state=0).fit(X)
    correct, incorrect_names = accuracy(model.labels_, y, instance_names)
    return model, correct, incorrect_names

def accuracy(y_test, y_true, instance_names, strategy_names=None):
    correct = 0
    incorrect_names = []
    n = len(y_true)
    for i in range(n):
        if (int(y_test[i]) == int(y_true[i])):
            correct += 1
        else:
            incorrect_names.append(instance_names[y_true[i]])
            if strategy_names:
                print("Incorrected guessed {0} instead of {1}".format(\
                    strategy_names[y_test[i]], strategy_names[y_true[i]]))
    return (correct * 1.0)/n, incorrect_names

def linear_regression(X, labels, instance_names, cross_val=False): #93.5%
    # reg = LinearRegression()
    # loocv = LeaveOneOut()
    if cross_val:
        for i in range(2, 9):
            print("CV:", i)
            reg = LinearRegression()
            results_loocv = cross_val_score(reg, X, labels, cv=i)
            print(results_loocv)
            print("mean R^2:", results_loocv.mean())
    
    reg = LinearRegression().fit(X, labels)
    score = reg.score(X, labels)
    pred_label = reg.predict(X)
    correct, incorrect_names = accuracy(pred_label, labels, instance_names)
    return reg, correct, incorrect_names

def logistic_regression(X, y, instance_names, cross_val=False):
    if cross_val:
        for i in range(2, 9):
            print("CV:", i)
            clf = LogisticRegression()
            results_loocv = cross_val_score(clf, X, y, cv=i)
            print(results_loocv)
            print("mean R^2:", results_loocv.mean())

    clf = LogisticRegression(random_state=0).fit(X, y)
    score = clf.score(X, y)
    pred_label = clf.predict(X)
    correct, incorrect_names = accuracy(pred_label, y, instance_names)
    return clf, correct, incorrect_names

def svm(X, y, instance_names, strategy_names):
    clf = SVC(kernel='rbf',gamma='auto').fit(X, y)
    score = clf.score(X,y) 
    pred_label = clf.predict(X)
    correct, incorrect_names = accuracy(pred_label, y, instance_names, strategy_names)
    return clf, correct, incorrect_names

def dbscan(X, y, instance_names): #62.5%
    best_score = -1
    best_labels = None
    best_eps = None
    best_ms = None
    best_incorrect = []
    eps_a = np.arange(0.01, 2.0, 0.01)
    for eps in eps_a:
        for min_sample in range(1, 3):
            clustering = DBSCAN(eps=eps, min_samples=min_sample).fit(X)
            correct, incorrect_names = accuracy(clustering.labels_, y, instance_names)
            if correct > best_score:
                best_score = correct
                best_labels = clustering.labels_
                best_eps = eps
                best_ms = min_sample
                best_incorrect = incorrect_names
    return best_score, best_incorrect

def pretty_print(best_performance):
    for instance_name, strat_info in best_performance.items():
        print(instance_name, strat_info[0])

def validate_solution(logname):

    def get_route(routes):
        start = None
        end = None
        paths = []
        path = []
        for idx, customer in enumerate(routes):
            if customer == 0 and not start and not end:
                start = True
                path.append(customer)
            elif customer == 0 and not end and start:
                end = True
                path.append(customer)
            elif start and not end:
                path.append(customer)
            if start and end:
                start = None
                end = None
                paths.append(path)
                path = []
        return paths

    def check_num_vehicles(routes, num_vehicles):
        assert num_vehicles == len(routes), \
        "number of vehicles is {0} and routes length is {1}".format(num_vehicles, len(routes))

    def check_unique_passengers(routes, numCustomers):
        customers_visited = set()
        for route in routes:
            for customer in route:
                if customer == 0: 
                    customers_visited.add(customer)
                    continue

                assert customer not in customers_visited, \
                "customers are not unique: {0} visited again".format(customer)

                customers_visited.add(customer)
        # assert len(customers_visited) == numCustomers-1, "not all customers were visited"
        if len(customers_visited) != numCustomers:
            print(len(customers_visited), numCustomers)
            print("not all customers were visited")
        for i in range(numCustomers):
            if i not in customers_visited:
                print(i)
    def check_vehicle_starts_and_ends_at_depo(routes):
        for route in routes:
            assert route[0] == 0 and route[-1] == 0, "Vehicle must start and end at depo"

    def calculate_distance(x1, y1, x2, y2):
        return (((x1-x2)*(x1-x2))+((y1-y2)*(y1-y2)))**0.5

    def calculate_total_distance(x, y, routes, demand, reported_distance):
        truck_x, truck_y = x[0], y[0]
        total_distance = 0.0
        for route in routes:
            for customer in route:
                customer_x, customer_y = x[customer], y[customer]
                total_distance += calculate_distance(truck_x, truck_y, customer_x, customer_y)
                truck_x, truck_y = customer_x, customer_y
        total_distance = round(total_distance, 2)
        if total_distance != reported_distance:
            print("Total distance is incorrect")
            print(total_distance, reported_distance)
        # assert reported_distance == total_distance, "Total distance is incorrect"

    def check_vehicle_capacity(routes, vehicle_capacity, demand):
        for route in routes:
            capacity = vehicle_capacity
            for customer in route:
                capacity -= demand[customer]
            assert capacity >= 0, "Max vehicle capacity was exceeded"

    def get_instance_information(instance_name):
        with open("input/{0}".format(instance_name), 'r') as f:
            lines = f.readlines()
        numCustomers, numVehicles, vehicleCapacity = [int(x) for x in lines[0].split()]
        demand = [] #the demand of each customer
        x = [] #the x coordinate of each customer
        y = [] #the y coordinate of each customer
        for idx, line in enumerate(lines[1:]):
            line = line.split()
            if line:
                demand.append(float(line[0]))
                x.append(float(line[1]))
                y.append(float(line[2]))
        return numCustomers, numVehicles, vehicleCapacity, demand, x, y


    instances = {}
    with open(logname, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()
        instance_name = line[1].split('/')
        if len(instance_name) > 1:
            instance_name = instance_name[-1]
        else:
            instance_name = instance_name[0]
        distance = line[5]
        path = [int(x) for x in line[8:]]
        instances[instance_name] = (distance, path)

    for instance_name, properties in instances.items():
        # if instance_name == '135_7_1.vrp':
        print(instance_name)
        distance = float(properties[0])
        routes = properties[1]
        numCustomers, numVehicles, vehicleCapacity, demand, x, y = get_instance_information(instance_name)
        routes = get_route(routes)
        check_num_vehicles(routes, numVehicles)
        check_unique_passengers(routes, numCustomers)
        check_vehicle_starts_and_ends_at_depo(routes)
        check_vehicle_capacity(routes, vehicleCapacity, demand)
        calculate_total_distance(x, y, routes, demand, distance)

def get_logs():
    path = 'logs/'
    files = (file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)) and '.log' in file)
    return files


def write_final_results_to_csv():
    files = get_logs()
    strategies = []
    for file in files: 
        log_name = os.path.join('logs', file)
        parts = log_name.split('_')[1:]
        strategy_name = ''
        for part in parts:
            if part == "final" or "log" in part:
                break
            strategy_name += part
        strategy = StrategyInfo(strategy_name, log_name)
        strategies.append(strategy)
        df_distance = pd.DataFrame(strategy.performance,index=["distance"])
        df_time = pd.DataFrame(strategy.time,index=["time"])
        df = pd.concat([df_distance, df_time])
        df.to_csv(os.path.join('results',strategy_name+'.csv'))
    return strategies
def format_output_to_generate_csv():    
    instance_names = ["101_11_2.vrp", "101_8_1.vrp", "121_7_1.vrp", "135_7_1.vrp", \
    "151_15_1.vrp", "16_5_1.vrp", "200_16_2.vrp", "21_4_1.vrp", "241_22_1.vrp", "262_25_1.vrp", \
    "30_4_1.vrp", "386_47_1.vrp","41_14_1.vrp", "45_4_1.vrp", "51_5_1.vrp", "76_8_2.vrp"]
    instance_to_instaceIDs = {name : i+1 for i, name in enumerate(instance_names)}
    # headers = ["screenname", "instanceID", "yPlot", "instance", "time", "result"]
    files = get_logs()
    output = "screenname,instanceID,yPlot,instance,time,result\n"
    for file in files: 
        log_name = os.path.join('logs', file)
        parts = log_name.split('_')[1:]
        strategy_name = ''
        for part in parts:
            if part == "final" or "log" in part:
                break
            strategy_name += part
        strategy = StrategyInfo(strategy_name, log_name)
        for instance in instance_names:
            distance = strategy.performance[instance]
            time = strategy.time[instance]
            output += '{0},{1},{2},{3},{4},{5}'.format(strategy_name,str(instance_to_instaceIDs[instance]), \
                                                str(distance), instance, str(time), str(distance))
            output += '\n'
    #with open("p5_visualization_website/aggregated_results.csv", 'w') as f:
    with open("aggregated_results.csv", 'w') as f:
        f.write(output)

def calculate_distance_by_truck():
    pass

def validate_logs():
    files = get_logs()
    for file in files:
        validate_solution(file)

def get_best_performers(strategies):
    best_performance = {instance_name : (strategies[0].name, distance) for instance_name, distance in strategies[0].performance.items()}
    times = {strategy.name : strategy.time for strategy in strategies}
    strategy_names = []
    for strategy in strategies:
        strategy_names.append(strategy.name)
        for instance_name, distance in strategy.performance.items():
            if best_performance[instance_name][1] > distance:
                best_performance[instance_name] = (strategy.name, distance)
            elif best_performance[instance_name][1] == distance:
                if times[strategy.name][instance_name] > times[best_performance[instance_name][0]][instance_name]:
                    best_performance[instance_name] = (strategy.name, distance)
    return best_performance, strategy_names

def create_labels(best_performance, strategy_names):
    del strategy_names[1] #doesn't appear in labels
    name_to_label = {name : i for i, name in enumerate(strategy_names)}
    X = []
    y = []
    instance_names = []
    for instance_name, pair in best_performance.items():
        strategy_name = pair[0]
        print(get_instance_details(instance_name))
        X.append(get_instance_details(instance_name))
        y.append(name_to_label[strategy_name])
        instance_names.append(instance_name)
    return X, y, strategy_names, instance_names, name_to_label

def reshape_array(Z, shape):
    Z = Z.reshape(shape)
    Z = Z.reshape((Z.shape[0]*Z.shape[1]), Z.shape[2])
    Z = Z.transpose()
    return Z
def plot_decision_regions(X,y,classifier,strategies,test_idx=None,resolution=0.02):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Initialise the marker types and colors
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    color_Map = ListedColormap(colors[:len(np.unique(y))]) #we take the color mapping correspoding to the 
                                                            #amount of classes in the target data
    
    # Parameters for the graph and decision surface
    # RANGE = 1
    # x1_min = X[:,0].min() - RANGE
    # x1_max = X[:,0].max() + RANGE
    # x2_min = X[:,1].min() - RANGE
    # x2_max = X[:,1].max() + RANGE
    # x3_min = X[:,2].min() - RANGE
    # x3_max = X[:,2].max() + RANGE
    # print(x1_min, x1_max)
    # print(x2_min, x2_max)
    # print(x3_min, x3_max)
    # xx1, xx2, xx3 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
    #                        np.arange(x2_min,x2_max,resolution),
    #                        np.arange(x3_min, x3_max,resolution))
    # xx1, xx2 = np.meshgrid(X, y)
    # Z = classifier.predict(X)
    # print(np.unique(Z))
    # Z = classifier.decision_function(X)

    # shape = xx1.shape
    # xx1 = reshape_array(xx1, shape)
    # xx2 = reshape_array(xx2, shape)
    # Z = reshape_array(Z, shape)
    # Z = Z.reshape((8,2))
    # print(np.unique(Z))
    # c=np.array([color_Map])
    # ax.contourf(xx1,xx2, Z, alpha=0.4,cmap=color_Map)
    # plt.xlim(xx1.min(),xx1.max())
    # plt.ylim(xx2.min(),xx2.max())
    # plt.zlim(xx3.min(), xx3.max())
    
    # Plot samples
    # X_test, Y_test = X[test_idx,:], y[test_idx]
    # ax.scatter(xx1, xx2, xx3, Z,
    #               alpha = 0.8, c = [color_Map],
    #                )
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(xs = X[y == cl, 0], ys = X[y == cl, 1], \
                    zs = X[y == cl, 2], \
                    alpha = 0.8, c = [color_Map(idx)],
                    marker = markers[idx], label = strategies[cl].name
                   )

    ax.legend()
    plt.show()

def graph_svm(model, X, y, strategies):
    X = np.array(X)
    y = np.array(y)
    # sc = StandardScaler()
    # X = sc.fit_transform(X)
    plot_decision_regions(X, y, model, strategies)
    # ax = plt.gca()
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # xlim = [X[:,0].min() - 1, X[:,0].max() + 1]
    # ylim = [X[:,1].min() - 1, X[:,1].max() + 1]
    # # create grid to evaluate model
    # xx = np.linspace(xlim[0], xlim[1], 30)
    # yy = np.linspace(ylim[0], ylim[1], 30)
    # YY, XX = np.meshgrid(yy, xx)
    # xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Z = model.decision_function(np.array([XX.ravel(), YY.ravel()]))#.reshape(XX.shape)
    # # plot decision boundary and margins
    # print(Z.shape)
    # ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
    #            linestyles=['--', '-', '--'])
    # # plot support vectors
    # ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
    #            linewidth=1, facecolors='none', edgecolors='k')
    # plt.show()

def graph_linear_svm(clf, X, y):
    X = np.array(X)
    y = np.array(y)
    h = .02  # step size in the mesh

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel']

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('number of vehicles')
    plt.ylabel('number of customers')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[0])

    plt.show()


def graph_pca(models, U, v):
    for model in models:
        pca = PCA(n_components=2)
        pca.fit(U,v)
        U2 = pca.fit_transform(U,v)
        model.fit(U2,v)

        # generate grid for plotting
        h = 0.2
        x_min, x_max = U2[:,0].min() - 1, U2[:, 0].max() + 1
        y_min, y_max = U2[:,1].min() - 1, U2[:, 1].max() + 1
        xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h))

        # create decision boundary plot
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx,yy,Z,cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(U2[:,0],U2[:,1],c=v)
        plt.show()

def main():
    strategies = write_final_results_to_csv()
    format_output_to_generate_csv()
    best_performance, strategy_names = get_best_performers(strategies)
    # for strategy in strategies:
    #     if strategy.name == "evo2optsa":
    #         print(strategy.name)
    #         print(strategy.performance['121_7_1.vrp'], best_performance['121_7_1.vrp'][1])
    #         print(strategy.time['121_7_1.vrp'])
    #         print(strategy.time)
    #     if strategy.name == "2opt":
    #         print(strategy.name)
    #         print(strategy.time['121_7_1.vrp'])
    #         print(strategy.performance['121_7_1.vrp'], best_performance['121_7_1.vrp'][1])
    X, labels, strategy_names, instance_names, name_to_label = create_labels(best_performance, strategy_names)

    # models = []
    label_to_name = {v:k for k,v in name_to_label.items()}
    model, correct, incorrect_names = svm(X, labels, instance_names, strategy_names)
    print(correct)
    dump(model, 'decision_boundary.joblib')
    with open('decision_boundary.sav', 'wb') as f:
        pickle.dump(model, f)
    with open("label_to_name.json", 'w') as f:
        json.dump(label_to_name, f)
    # models.append(model)
    # model1, correct, incorrect_names = KNN(X, labels, instance_names)
    # models.append(model1)
    # model2, correct, incorrect_names = kmeans(X, labels, instance_names)
    # models.append(model2)
    # model3, correct, incorrect_names = linear_regression(X, labels, instance_names)
    # models.append(model3)
    # model4, correct, incorrect_names = logistic_regression(X, labels, instance_names)
    # models.append(model4)
    # model, correct, incorrect_names = randomforest(X, labels, instance_names)
    # models.append(model)
    # graph_svm(model, X, labels, strategies)
    # graph_pca(models, X, labels)
    

if __name__ == '__main__':
    
    # validate_logs()
    validate_solution("results3.log")
    #main()





