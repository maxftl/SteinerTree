\ Model Network optimization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  mu[0] + mu[1] + mu[2]
Subject To
 R0: - 2 f[0,0] + mu[0] >= 0
 R1: - 2 f[0,1] + mu[0] >= 0
 R2: - 2.23606797749979 f[1,0] + mu[1] >= 0
 R3: - 2.23606797749979 f[1,1] + mu[1] >= 0
 R4: - 2.23606797749979 f[2,0] + mu[2] >= 0
 R5: - 2.23606797749979 f[2,1] + mu[2] >= 0
 R6: f[0,0] + f[1,0] = 1
 R7: f[0,1] + f[1,1] = 0
 R8: - f[0,0] + f[2,0] = 0
 R9: - f[0,1] + f[2,1] = 1
 R10: - f[1,0] - f[2,0] = -1
 R11: - f[1,1] - f[2,1] = -1
Bounds
End
