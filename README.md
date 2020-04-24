# vehicle_routing_problem
To run a single test file: python runner.py -s 5_4_10.vrp

To run all the test files: python runner.py

To specify the type of strategy: python runner.py -a random

So far the following strategies are available:
- random: Randomly choose search strategy
- random_greedy: order search by customer demand

To test different seeds: python runner.py -a <strategy> -t
- random: best seed is 0 when tested from 0 to 10,000 across all the test files
- random_greedy: did not finish testing in 5 minutes; needs to run for longer to find out