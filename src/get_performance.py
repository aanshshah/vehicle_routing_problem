def determine_leaderboard_performance(filepath):
    performance = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        line = line.split(',')
        screenname = line[0]
        distance = float(line[2])
        instance_name = line[3]
        if performance.get(instance_name, (None, float('inf')))[1] > distance:
            performance[instance_name] = (screenname, distance)
    return performance

def get_my_best_performance(filepath, my_screenname):
    leaderboard_performance = determine_leaderboard_performance(filepath)
    counter = 0
    for instance_name, (screenname, _) in leaderboard_performance.items():
        if screenname == my_screenname: counter += 1
    return counter

def main():
    my_screenname = "036ebde0"
    filepath = "/course/cs2951o/www/p5aggregate.csv"
    print(get_my_best_performance(filepath, my_screenname))

if __name__ == '__main__':
    main()