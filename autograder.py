import argparse
import subprocess
import pandas as pd
import numpy as np

seeds = [747, 374]
numTracks = 6
baseline0_scores = [[339, 343, 337, 342, 338, 316], [342, 345, 338, 343, 340, 318]]  # Conservative baseline: 2 lists correspond to 2 seeds
baseline1_scores = [[517, 593, 537, 577, 590, 293], [546, 637, 550, 629, 598, 305]]  # Aggressive baseline: 2 lists correspond to 2 seeds


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cmaes", action='store_true')
    args = parser.parse_args()
    used_cmaes = args.cmaes

    marks = [[0 for i in range(numTracks)], [0 for i in range(numTracks)]]  # 2 lists correspond to 2 seeds
    seed_marks = []

    for s, seed in enumerate(seeds):
        print('Running for seed', seed)
        print()

        if used_cmaes:
            cmd_planner = "python", "main.py", "--eval", "--numTracks", str(numTracks), "--seed", str(seed)
        else:
            cmd_planner = "python", "main.py", "--numTracks", str(numTracks), "--seed", str(seed)

        subprocess.check_output(cmd_planner, universal_newlines=True)

        perf_df = pd.read_json("Performance_" + str(seed) + ".json")
        perf_scores = perf_df["Score"].tolist()

        for t in range(numTracks):
            if perf_scores[t] >= baseline0_scores[s][t]:
                marks[s][t] += 0.5
            if perf_scores[t] >= baseline1_scores[s][t]:
                marks[s][t] += 0.5
            print("Track", t, "marks:", marks[s][t], "out of 1")

        seed_marks.append(np.sum(marks[s]))
        print()
        print("Seed", seed, "marks:", seed_marks[s], "out of", numTracks)
        print("*" * 100)


    print("Total marks:", np.sum(seed_marks), "out of", len(seeds)*numTracks)
    print("Scaled marks:", np.round((np.sum(seed_marks)/3.0), 2), "out of 4")