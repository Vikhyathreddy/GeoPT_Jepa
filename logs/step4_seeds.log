Question-1 branch (decided by E1-c', reported first): **branch 2: close Q1 as not testable with this latent design**

| id | task | claim | per-seed values (s0..s4) | in direction | across-seed mean [95% t interval] | verdict |
|---|---|---|---|---|---|---|
| E1-c' | A | clean LT beats clean N: err(LT)/err(N) - 1 (decides Q1 branch) | +0.0321, -0.0772, +0.0889, -0.0047, -0.0887 | 3/5 | -0.0099 [-0.1026, +0.0828] | single-seed |
| E1-c' | B | clean LT beats clean N: err(LT)/err(N) - 1 (decides Q1 branch) | +0.0070, -0.0521, -0.0518, -0.0126, -0.0184 | 4/5 | -0.0256 [-0.0577, +0.0065] | single-seed |
| E1-a | A | clean L beats clean LT: err(LT)/err(L) - 1 | +0.3152, +0.0908, +0.1023, +0.2862, +0.0828 | 5/5 | +0.1754 [+0.0327, +0.3182] | robust |
| E1-b | A | clean L beats N: err(N)/err(L) - 1 | +0.2889, +0.1880, +0.0205, +0.3041, +0.1944 | 5/5 | +0.1992 [+0.0588, +0.3396] | robust |
| E1-d | A | label sigma 2 degrades L: err(L,s2)/err(L) - 1 | +0.1230, +0.1109, +0.0839, +0.1647, +0.0450 | 5/5 | +0.1055 [+0.0501, +0.1609] | robust |
| E1-e | A | at sigma 2 L beats clean LT: err(LT)/err(L,s2) - 1 | +0.1900, -0.0104, +0.0315, +0.1173, +0.0428 | 4/5 | +0.0742 [-0.0244, +0.1729] | single-seed |
| E1-f | A | at sigma 4 L beats clean LT: err(LT)/err(L,s4) - 1 | +0.0959, -0.0775, -0.0221, -0.0195, -0.1192 | 1/5 | -0.0285 [-0.1290, +0.0721] | single-seed |
| E1-a | B | clean L beats clean LT: err(LT)/err(L) - 1 | +0.0730, +0.1499, +0.0666, +0.1264, +0.1077 | 5/5 | +0.1047 [+0.0609, +0.1485] | robust |
| E1-b | B | clean L beats N: err(N)/err(L) - 1 | +0.0734, +0.2253, +0.1328, +0.1446, +0.1390 | 5/5 | +0.1430 [+0.0758, +0.2103] | robust |
| E1-d | B | label sigma 2 degrades L: err(L,s2)/err(L) - 1 | +0.0494, +0.1323, +0.0677, +0.1097, +0.0736 | 5/5 | +0.0865 [+0.0447, +0.1283] | robust |
| E1-e | B | at sigma 2 L beats clean LT: err(LT)/err(L,s2) - 1 | +0.0280, +0.0248, +0.0072, +0.0247, +0.0378 | 5/5 | +0.0245 [+0.0108, +0.0383] | robust |
| E2-a | A n=64 | zero prompt hurts L: err(zero) - err(0) | +0.1926, +0.1886, +0.2021, +0.1628, +0.1817 | 5/5 | +0.1855 [+0.1673, +0.2038] | robust |
| E2-d | A n=64 | L loses more than scratch at 90: diff-in-diff | -0.0320, -0.0214, -0.0242, -0.0546, -0.0316 | 0/5 | -0.0328 [-0.0490, -0.0166] | contradicted |
| E2-g | A n=64 | 90 helps L (own delta): err(90) - err(0) | -0.0293, -0.0188, -0.0216, -0.0520, -0.0289 | 5/5 | -0.0301 [-0.0463, -0.0139] | robust |
| E2-a | B n=64 | zero prompt hurts L: err(zero) - err(0) | +0.2816, +0.2728, +0.2942, +0.2965, +0.2831 | 5/5 | +0.2857 [+0.2736, +0.2977] | robust |
| E2-d | B n=64 | L loses more than scratch at 90: diff-in-diff | -0.0327, -0.0281, -0.0190, -0.0173, -0.0158 | 0/5 | -0.0226 [-0.0318, -0.0133] | contradicted |
| E2-g | B n=64 | 90 helps L (own delta): err(90) - err(0) | -0.0329, -0.0283, -0.0192, -0.0174, -0.0160 | 5/5 | -0.0227 [-0.0320, -0.0135] | robust |
| E2-b | B n=64 | reversed prompt beats aligned for L: err(180) - err(0) | -0.0202, -0.0349, -0.0261, -0.0187, -0.0159 | 5/5 | -0.0231 [-0.0325, -0.0138] | robust |
| E2-c | A n=64 | 180 costs about nothing for L (null-type) | -0.0024, -0.0068, +0.0261, -0.0186, -0.0076 | - | -0.0018 [-0.0226, +0.0189] (2 SE_seed = 0.0235) | holds |
| E2-a | A n=16 | zero prompt hurts L: err(zero) - err(0) | +0.1235, +0.1061, +0.1220, +0.0922, +0.1159 | 5/5 | +0.1119 [+0.0958, +0.1280] | robust |
| E2-d | A n=16 | L loses more than scratch at 90: diff-in-diff | +0.0164, -0.0011, +0.0142, -0.0177, +0.0059 | 3/5 | +0.0035 [-0.0136, +0.0206] | single-seed |
| E2-g | A n=16 | 90 helps L (own delta): err(90) - err(0) | +0.0163, -0.0012, +0.0141, -0.0178, +0.0058 | 2/5 | +0.0035 [-0.0136, +0.0205] | single-seed |
| E2-a | B n=16 | zero prompt hurts L: err(zero) - err(0) | +0.0126, +0.0218, +0.0174, +0.0089, +0.0228 | 5/5 | +0.0167 [+0.0093, +0.0241] | robust |
| E2-d | B n=16 | L loses more than scratch at 90: diff-in-diff | -0.0521, -0.0062, -0.0732, -0.0321, -0.0538 | 0/5 | -0.0435 [-0.0751, -0.0119] | contradicted |
| E2-g | B n=16 | 90 helps L (own delta): err(90) - err(0) | -0.0538, -0.0079, -0.0750, -0.0338, -0.0555 | 5/5 | -0.0452 [-0.0768, -0.0136] | robust |
| E2-b | B n=16 | reversed prompt beats aligned for L: err(180) - err(0) | -0.1020, -0.1207, -0.1155, -0.1076, -0.0777 | 5/5 | -0.1047 [-0.1254, -0.0840] | robust |
| E2-c | A n=16 | 180 costs about nothing for L (null-type) | +0.0099, -0.0040, +0.0036, -0.0018, +0.0101 | - | +0.0036 [-0.0045, +0.0117] (2 SE_seed = 0.0397) | holds |
| E3-a | A | L ranks better than LT, decoder probe: rho(L) - rho(LT) | -0.0032, +0.0016, +0.0089, +0.0126, -0.0161 | 3/5 | +0.0007 [-0.0132, +0.0147] | single-seed |
| E3-b | A | L ranks better than LT, R1 (ft seed 0): rho(L) - rho(LT) | +0.0342 | - | - | single-seed (R1 not replicated by design, 2026-10-21) |
| E3-a | B | L ranks better than LT, decoder probe: rho(L) - rho(LT) | +0.0144, +0.0219, +0.0014, +0.0056, +0.0079 | 5/5 | +0.0102 [+0.0002, +0.0202] | robust |
| E3-b | B | L ranks better than LT, R1 (ft seed 0): rho(L) - rho(LT) | +0.0034 | - | - | single-seed (R1 not replicated by design, 2026-10-21) |
