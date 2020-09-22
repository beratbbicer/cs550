import numpy as np
from sklearn.tree import DecisionTreeClassifier as DecisionTree
import itertools

def get_data(train_file="train.txt", test_file="test.txt", cost_file="cost.txt"):
    raw_train = np.loadtxt(train_file)
    x_train, y_train = raw_train[:,:-1], raw_train[:,-1]
    raw_test = np.loadtxt(test_file)
    x_test, y_test = raw_test[:,:-1], raw_test[:,-1]
    costs = np.genfromtxt(cost_file, dtype="str")
    feature_names = np.hstack((costs[:,0], np.array("fused_feature")))
    feature_selection_costs = costs[:,1].astype(np.float)
    return x_train, x_test, y_train, y_test, feature_names, feature_selection_costs

def train_tree(x_train, y_train, feature_set):
    tree = DecisionTree(splitter="best", class_weight="balanced", min_samples_leaf=3, max_depth=8)
    tree.fit(x_train[:, feature_set], y_train)
    return tree

def classwise_accuracy(preds, y_test):
    indices_1 = np.argwhere(y_test == 1)
    indices_2 = np.argwhere(y_test == 2)
    indices_3 = np.argwhere(y_test == 3)
    accuracy_1 = np.sum(preds[indices_1] == y_test[indices_1]) / len(y_test[indices_1])
    accuracy_2 = np.sum(preds[indices_2] == y_test[indices_2]) / len(y_test[indices_2])
    accuracy_3 = np.sum(preds[indices_3] == y_test[indices_3]) / len(y_test[indices_3])
    return accuracy_1, len(indices_1), accuracy_2, len(indices_2), accuracy_3, len(indices_3)

def get_joint_cost(accuracy_1, count_1, accuracy_2, count_2, accuracy_3, count_3, feature_set, feature_selection_costs):
    classification_success = accuracy_1 * 1 + accuracy_2 * 1 + accuracy_3 * 0.5
    classification_variation = abs(accuracy_1 - accuracy_2) + 0.5 * abs(accuracy_1 - accuracy_3) + 0.5 * abs(accuracy_2 - accuracy_3)

    new_feature_set = feature_set[(feature_set != 18) & (feature_set != 19) & (feature_set != 20)]
    feature_selection_cost = np.sum(feature_selection_costs[new_feature_set])
    
    if 20 in feature_set:
        feature_selection_cost += feature_selection_costs[18] + feature_selection_costs[19]
    else:
        if 18 in feature_set:
            feature_selection_cost += feature_selection_costs[18]

        if 19 in feature_set:
            feature_selection_cost += feature_selection_costs[19]

    joint_cost = feature_selection_cost * 0.0175 + classification_variation - classification_success * 2
    return joint_cost

def forward_selection(x_train, x_test, y_train, y_test, all_feature_set, feature_selection_costs, feature_names):
    best_cost = 10**9
    current_feature_set = []
    best_tree = None
    while len(current_feature_set) < len(all_feature_set):
        costs = []
        features_tried = []
        accuracies_1 = []
        accuracies_2 = []
        accuracies_3 = []
        overall_accuracies = []
        new_feature = []
        best_trees = []
        for feature in all_feature_set:
            if feature in current_feature_set:
                continue
            
            new_feature_set = current_feature_set + [feature]
            tree = train_tree(x_train, y_train, new_feature_set)
            preds = tree.predict(x_train[:, new_feature_set])
            accuracy_1, count_1, accuracy_2, count_2, accuracy_3, count_3 = classwise_accuracy(preds, y_train)
            joint_cost = get_joint_cost(accuracy_1, count_1, accuracy_2, count_2, accuracy_3, count_3, 
                np.array(new_feature_set, dtype = np.int32), feature_selection_costs)
            costs.append(joint_cost)
            features_tried.append(new_feature_set)
            accuracies_1.append(accuracy_1)
            accuracies_2.append(accuracy_2)
            accuracies_3.append(accuracy_3)
            overall_accuracies.append(np.sum(preds == y_train) / len(y_train))
            new_feature.append(feature)
            best_trees.append(tree)
        
        best_set_index = costs.index(min(costs))
        if costs[best_set_index] < best_cost:
            print("\nNew feature: " + str(new_feature[best_set_index]))
            print("Accuracies: %.3f %.3f %.3f %.3f" % (overall_accuracies[best_set_index], accuracies_1[best_set_index], accuracies_2[best_set_index], accuracies_3[best_set_index]))
            print("Best Cost: %.3f" % (costs[best_set_index]))
            current_feature_set = features_tried[best_set_index]
            best_cost = costs[best_set_index]
            best_tree = best_trees[best_set_index]
        else:
            break

    preds_train = best_tree.predict(x_train[:, current_feature_set])
    preds_test = best_tree.predict(x_test[:, current_feature_set])   
    train_accuracy_1, _, train_accuracy_2, _, train_accuracy_3, _ = classwise_accuracy(preds_train, y_train)
    test_accuracy_1, _, test_accuracy_2, _, test_accuracy_3, _ = classwise_accuracy(preds_test, y_test)

    print("\nTrain Accuracies: %.3f %.3f %.3f %.3f" % (np.sum(preds_train == y_train) / len(y_train), train_accuracy_1, train_accuracy_2, train_accuracy_3))
    print("Test Accuracies: %.3f %.3f %.3f %.3f" % (np.sum(preds_test == y_test) / len(y_test), test_accuracy_1, test_accuracy_2, test_accuracy_3))
    print("Selected features: " + str(feature_names[current_feature_set]))
    print("Cost: %.3f" % (best_cost))
    return best_tree, current_feature_set, best_cost

def breed(parents):
    np.random.shuffle(parents)
    parents_1 = parents[:int(len(parents)/2)]
    parents_2 = parents[int(len(parents)/2):]
    cross_points = np.random.choice(np.array(list(range(21))), 5)
    cross_points = np.sort(cross_points)
    offspring = []

    for i in range(len(parents_1)):
        child_1, child_2  = [c for c in parents_1[i]], [c for c in parents_2[i]]
        for j in range(1,len(cross_points)):
            tmp = child_1[cross_points[j-1]:cross_points[j]]
            child_1[cross_points[j-1]:cross_points[j]] = child_2[cross_points[j-1]:cross_points[j]]
            child_2[cross_points[j-1]:cross_points[j]] = tmp
        
        t = np.char.find(child_1, "1")
        if len(t[t == 0]) == 0:  
            child_1 = [c for c in parents_1[i]]
        
        t = np.char.find(child_2, "1")
        if len(t[t == 0]) == 0:
            child_2 = [c for c in parents_2[i]]
        
        offspring.extend(["".join(child_1), "".join(child_2)])
    return offspring

def mutate(population, count):
    indices = np.random.choice(len(population), count, replace=False)
    mutation_points = np.random.choice(np.array(list(range(21))), 5)
    for i in indices:
        individual = [c for c in population[i]]

        for j in mutation_points:
            if individual[j] == '1':
                individual[j] = '0'
            else:
                individual[j] = '1'
        
        t = np.char.find(individual, "1")
        if len(t[t == 0]) == 0:  
            individual = [c for c in population[i]]

        population[i] = "".join(individual)

def genetic_selection(x_train, x_test, y_train, y_test, feature_selection_costs, feature_names):
    all_features = np.array(list(range(21)))
    gen_size = 2500
    selection_size = int(gen_size/5)
    tmp = ["".join(seq) for seq in itertools.product("01", repeat=21) if seq.count("1") <= 7 and seq.count("1") > 0]    
    all_comb = np.array(tmp)
    population = np.array(all_comb[np.random.choice(len(all_comb), gen_size, replace=False)])
    best_score = 999999
    
    while True:
        joint_costs, classifiers, next_gen, accuracies_1, accuracies_2, accuracies_3, overall_accuracies, next_gen = [], [], [], [], [], [], [], []
        for f in population:
            features = all_features[np.argwhere(np.array(list(f), dtype=np.int32).flatten() == 1).flatten()]
            tree = train_tree(x_train, y_train, features)
            preds = tree.predict(x_train[:, features])
            accuracy_1, count_1, accuracy_2, count_2, accuracy_3, count_3 = classwise_accuracy(preds, y_train)
            joint_costs.append(get_joint_cost(accuracy_1, count_1, accuracy_2, count_2, accuracy_3, count_3, features, feature_selection_costs))
            classifiers.append(tree)
            accuracies_1.append(accuracy_1)
            accuracies_2.append(accuracy_2)
            accuracies_3.append(accuracy_3)
            overall_accuracies.append(np.sum(preds == y_train) / len(y_train))

        joint_costs, classifiers, accuracies_1, accuracies_2, accuracies_3, overall_accuracies = np.array(joint_costs), np.array(classifiers), np.array(accuracies_1), np.array(accuracies_2), np.array(accuracies_3), np.array(overall_accuracies)
        order = joint_costs.argsort()
        population, joint_costs, classifiers = population[order], joint_costs[order], classifiers[order]
        accuracies_1, accuracies_2, accuracies_3, overall_accuracies = accuracies_1[order], accuracies_2[order], accuracies_3[order], overall_accuracies[order]
        new_best = joint_costs[0]

        if best_score > new_best:
            print("\nFeatures: " + str(population[0]))
            print("Accuracies: %.3f %.3f %.3f %.3f" % (overall_accuracies[0], accuracies_1[0], accuracies_2[0], accuracies_3[0]))
            print("Best Cost: %.3f" % (joint_costs[0]))
            next_gen.extend(population[:selection_size])
            next_gen.extend(breed(population[np.random.choice(len(population), len(population) - selection_size, replace=False)]))
            k = len(next_gen)
            q = len(tmp)
            mutate(next_gen, int(gen_size/4))
            population = np.array(next_gen)
            best_score = new_best
            best_tree = classifiers[0]
            best_features = population[0]
        else:
            break
    t = best_features
    best_features = all_features[np.argwhere(np.array(list(best_features), dtype=np.int32).flatten() == 1).flatten()]
    preds_train = best_tree.predict(x_train[:, best_features])
    preds_test = best_tree.predict(x_test[:, best_features])   
    train_accuracy_1, _, train_accuracy_2, _, train_accuracy_3, _ = classwise_accuracy(preds_train, y_train)
    test_accuracy_1, _, test_accuracy_2, _, test_accuracy_3, _ = classwise_accuracy(preds_test, y_test)

    print("\nTrain Accuracies: %.3f %.3f %.3f %.3f" % (np.sum(preds_train == y_train) / len(y_train), train_accuracy_1, train_accuracy_2, train_accuracy_3))
    print("Test Accuracies: %.3f %.3f %.3f %.3f" % (np.sum(preds_test == y_test) / len(y_test), test_accuracy_1, test_accuracy_2, test_accuracy_3))
    print("Selected features: " + str(feature_names[best_features]))
    print("Cost: %.3f" % (joint_costs[0]))
    return classifiers[0], population[0], best_score

x_train, x_test, y_train, y_test, feature_names, feature_selection_costs = get_data()
print("Forward Selection")
classifier, features_selected, cost = forward_selection(x_train, x_test, y_train, y_test, np.array(list(range(21))), feature_selection_costs, feature_names)
print("\nGenetic Selection")
classifier, features_selected, cost = genetic_selection(x_train, x_test, y_train, y_test, feature_selection_costs, feature_names)