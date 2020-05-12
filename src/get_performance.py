def determine_leaderboard_performance(filepath):
    performance = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        line = line.split(',')
        screenname = line[0]
        distance = float(line[2])
        instance_name = line[3]
        if performance.get(instance_name, (None, float('inf')))[1] > distance and distance > 0:
            performance[instance_name] = (screenname, distance)
    return performance

def get_my_best_performance(leaderboard_performance, my_screenname):
    counter = 0
    for instance_name, (screenname, _) in leaderboard_performance.items():
        if screenname == my_screenname: counter += 1
    return counter

def get_all_other_best_performances(leaderboard_performance):
    # best = {screenname : 0 for instance_name, (screenname, _) in leaderboard_performance.items()}
    best = {}
    for instance_name, (screenname, _) in leaderboard_performance.items():
        best[screenname] = best.get(screenname, 0) + 1
    return best

def display_all_top_performers(best_performer_counts):
    sorted_counts = {k: v for k, v in sorted(best_performer_counts.items(), key=lambda item: item[1])}
    for screenname, count in sorted_counts.items():
        print("{0} has lowest distance in {1} instances".format(str(screenname), str(count)))

def main():
    my_screenname = "036ebde0"
    filepath = "/course/cs2951o/www/p5aggregate.csv"
    performance = determine_leaderboard_performance(filepath)
    counter = get_my_best_performance(performance, my_screenname)
    print("We have lowest distance in {0} instances".format(str(counter)))
    best_performers = get_all_other_best_performances(performance)
    display_all_top_performers(best_performers)


if __name__ == '__main__':
    main()
