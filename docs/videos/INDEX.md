# Rollout videos of the ablation runs

Each file is a 2x2 panel: passthrough and frozen on top, fused and oracle below. Every panel shows the command System 2 sends.
File name: `<scene id>__<start>__<seed>.mp4`. Outcome per condition: steps to success, or `fail`. Details in [../EXPERIMENTS.md](../EXPERIMENTS.md).

The three clips in [`docs/videos/`](.) used in the README (`libero_goal_bowl_on_stove_C.mp4`, `libero_10_stove_moka_B.mp4`, `drop_bowl_on_stove_C.mp4`) are hand-picked examples. The folders below hold every episode of the two runs.

## Run 17: all 10 libero_10 scenes, start B, 2 seeds

Folder: [libero_10_run17/](libero_10_run17/), 20 videos.

| Video | Task | Start | Seed | passthrough | frozen | fused | oracle |
|---|---|---|---|---|---|---|---|
| [e48a24b9__B__58717.mp4](libero_10_run17/e48a24b9__B__58717.mp4) | pick up the book and place it in the back compartment of the caddy | B | 58717 | 200 | fail | fail | 195 |
| [e48a24b9__B__58718.mp4](libero_10_run17/e48a24b9__B__58718.mp4) | pick up the book and place it in the back compartment of the caddy | B | 58718 | 187 | fail | 191 | fail |
| [0a341022__B__84262.mp4](libero_10_run17/0a341022__B__84262.mp4) | put both moka pots on the stove | B | 84262 | 560 | fail | fail | fail |
| [0a341022__B__84263.mp4](libero_10_run17/0a341022__B__84263.mp4) | put both moka pots on the stove | B | 84263 | 406 | fail | fail | fail |
| [d5d7fb20__B__1636.mp4](libero_10_run17/d5d7fb20__B__1636.mp4) | put both the alphabet soup and the cream cheese box in the basket | B | 1636 | fail | fail | fail | fail |
| [d5d7fb20__B__1637.mp4](libero_10_run17/d5d7fb20__B__1637.mp4) | put both the alphabet soup and the cream cheese box in the basket | B | 1637 | fail | 289 | fail | 351 |
| [90bacd07__B__61387.mp4](libero_10_run17/90bacd07__B__61387.mp4) | put both the alphabet soup and the tomato sauce in the basket | B | 61387 | 349 | 319 | fail | fail |
| [90bacd07__B__61388.mp4](libero_10_run17/90bacd07__B__61388.mp4) | put both the alphabet soup and the tomato sauce in the basket | B | 61388 | 358 | fail | fail | fail |
| [6de4e5d6__B__17690.mp4](libero_10_run17/6de4e5d6__B__17690.mp4) | put both the cream cheese box and the butter in the basket | B | 17690 | fail | 474 | 571 | 375 |
| [6de4e5d6__B__17691.mp4](libero_10_run17/6de4e5d6__B__17691.mp4) | put both the cream cheese box and the butter in the basket | B | 17691 | fail | fail | fail | fail |
| [19f6887a__B__87294.mp4](libero_10_run17/19f6887a__B__87294.mp4) | put the black bowl in the bottom drawer of the cabinet and close it | B | 87294 | 283 | fail | fail | fail |
| [19f6887a__B__87295.mp4](libero_10_run17/19f6887a__B__87295.mp4) | put the black bowl in the bottom drawer of the cabinet and close it | B | 87295 | 230 | 350 | fail | 348 |
| [d9a3a991__B__81749.mp4](libero_10_run17/d9a3a991__B__81749.mp4) | put the white mug on the left plate and put the yellow and white mug on the right plate | B | 81749 | 485 | 433 | 501 | fail |
| [d9a3a991__B__81750.mp4](libero_10_run17/d9a3a991__B__81750.mp4) | put the white mug on the left plate and put the yellow and white mug on the right plate | B | 81750 | fail | 414 | fail | fail |
| [a0c83050__B__74228.mp4](libero_10_run17/a0c83050__B__74228.mp4) | put the white mug on the plate and put the chocolate pudding to the right of the plate | B | 74228 | fail | fail | fail | fail |
| [a0c83050__B__74229.mp4](libero_10_run17/a0c83050__B__74229.mp4) | put the white mug on the plate and put the chocolate pudding to the right of the plate | B | 74229 | fail | fail | fail | fail |
| [ac3be9e1__B__7749.mp4](libero_10_run17/ac3be9e1__B__7749.mp4) | put the yellow and white mug in the microwave and close it | B | 7749 | fail | fail | fail | fail |
| [ac3be9e1__B__7750.mp4](libero_10_run17/ac3be9e1__B__7750.mp4) | put the yellow and white mug in the microwave and close it | B | 7750 | fail | fail | fail | fail |
| [a77d804e__B__20018.mp4](libero_10_run17/a77d804e__B__20018.mp4) | turn on the stove and put the moka pot on it | B | 20018 | fail | 190 | 191 | 186 |
| [a77d804e__B__20019.mp4](libero_10_run17/a77d804e__B__20019.mp4) | turn on the stove and put the moka pot on it | B | 20019 | fail | fail | fail | fail |

## Run 13: 3 seen scenes (2 libero_goal, 1 libero_10), starts C and B, 1 seed

Folder: [libero_goal_run13/](libero_goal_run13/), 4 videos.

| Video | Task | Start | Seed | passthrough | frozen | fused | oracle |
|---|---|---|---|---|---|---|---|
| [5996eef4__B__63896.mp4](libero_goal_run13/5996eef4__B__63896.mp4) | put the bowl on the stove | B | 63896 | 106 | fail | 104 | 104 |
| [5996eef4__C__63896.mp4](libero_goal_run13/5996eef4__C__63896.mp4) | put the bowl on the stove | C | 63896 | 92 | fail | 87 | 88 |
| [e946ecc2__C__39558.mp4](libero_goal_run13/e946ecc2__C__39558.mp4) | put the bowl on top of the cabinet | C | 39558 | 89 | fail | 98 | 100 |
| [a77d804e__C__20018.mp4](libero_goal_run13/a77d804e__C__20018.mp4) | turn on the stove and put the moka pot on it | C | 20018 | 200 | 218 | 210 | 206 |
