# intrinsic motivation experiments

SND model A :  8x8/4, 3x3/2, 3x3/1

SND model B :  3x3/2, 3x3/2, 3x3/2, 3x3/1


Name    |CND    |normalisation  | int reward coeff | regularisation coeff | contrastive loss | score    | reward  | 
--------|-------|---------------|------------------|----------------------|------------------|----------|---------|
0       | A     | mean, std     | 0.125            | 0.0                  | pair MSE         |7100      | 11      |
1       | A     | mean, std     | 0.25             | 0.0                  | pair MSE         |10200     | 15      |
2       | A     | mean, std     | 0.4              | 0.0                  | pair MSE         |10200     | 13  P   |
3       | A     | mean, std     | 0.5              | 0.0                  | pair MSE         |10500     | 13      |
4       | A     | mean, std     | 0.75             | 0.0                  | pair MSE         |N/A       | N/A P   |
5       | A     | mean, std     | 1.0              | 0.0                  | pair MSE         |N/A       | N/A P   |
10      | A     | none          | 0.125            | 0.0                  | pair MSE         |N/A       | N/A D   |
11      | A     | none          | 0.25             | 0.0                  | pair MSE         |N/A       | N/A D   |
12      | A     | none          | 0.4              | 0.0                  | pair MSE         |N/A       | N/A D   |

20      | B     | mean, std     | 0.25             | 0.0                  | pair MSE         |N/A       | N/A P   |
21      | B     | mean, std     | 0.5              | 0.0                  | pair MSE         |N/A       | N/A P   |
22      | B     | mean, std     | 0.75             | 0.0                  | pair MSE         |N/A       | N/A P   |
