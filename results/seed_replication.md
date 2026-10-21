Question-1 branch (decided by E1-c', reported first): **undecided (incomplete)**

| id | task | claim | per-seed values (s0..s4) | in direction | across-seed mean [95% t interval] | verdict |
|---|---|---|---|---|---|---|
| E1-c' | A | clean LT beats clean N: err(LT)/err(N) - 1 (decides Q1 branch) | +0.0321, -0.0772 | 1/2 | -0.0225 [-0.7163, +0.6713] | incomplete |
| E1-c' | B | clean LT beats clean N: err(LT)/err(N) - 1 (decides Q1 branch) | +0.0070, -0.0521 | 1/2 | -0.0226 [-0.3981, +0.3530] | incomplete |
| E1-a | A | clean L beats clean LT: err(LT)/err(L) - 1 | +0.3152, +0.0908 | 2/2 | +0.2030 [-1.2228, +1.6287] | incomplete |
| E1-b | A | clean L beats N: err(N)/err(L) - 1 | +0.2889, +0.1880 | 2/2 | +0.2384 [-0.4026, +0.8795] | incomplete |
| E1-d | A | label sigma 2 degrades L: err(L,s2)/err(L) - 1 | +0.1230, +0.1109 | 2/2 | +0.1169 [+0.0395, +0.1944] | incomplete |
| E1-e | A | at sigma 2 L beats clean LT: err(LT)/err(L,s2) - 1 | +0.1900, -0.0104 | 1/2 | +0.0898 [-1.1834, +1.3629] | incomplete |
| E1-f | A | at sigma 4 L beats clean LT: err(LT)/err(L,s4) - 1 | +0.0959 | - | - | incomplete |
| E1-a | B | clean L beats clean LT: err(LT)/err(L) - 1 | +0.0730, +0.1499 | 2/2 | +0.1115 [-0.3769, +0.5999] | incomplete |
| E1-b | B | clean L beats N: err(N)/err(L) - 1 | +0.0734, +0.2253 | 2/2 | +0.1494 [-0.8159, +1.1146] | incomplete |
| E1-d | B | label sigma 2 degrades L: err(L,s2)/err(L) - 1 | +0.0494, +0.1323 | 2/2 | +0.0908 [-0.4359, +0.6175] | incomplete |
| E1-e | B | at sigma 2 L beats clean LT: err(LT)/err(L,s2) - 1 | +0.0280, +0.0248 | 2/2 | +0.0264 [+0.0062, +0.0467] | incomplete |
| E2-a | A n=64 | zero prompt hurts L: err(zero) - err(0) | +0.1926, +0.1886 | 2/2 | +0.1906 [+0.1654, +0.2158] | incomplete |
| E2-d | A n=64 | L loses more than scratch at 90: diff-in-diff | -0.0320, -0.0214 | 0/2 | -0.0267 [-0.0938, +0.0404] | incomplete |
| E2-g | A n=64 | 90 helps L (own delta): err(90) - err(0) | -0.0293, -0.0188 | 2/2 | -0.0241 [-0.0911, +0.0430] | incomplete |
| E2-a | B n=64 | zero prompt hurts L: err(zero) - err(0) | +0.2816, +0.2728 | 2/2 | +0.2772 [+0.2214, +0.3331] | incomplete |
| E2-d | B n=64 | L loses more than scratch at 90: diff-in-diff | -0.0327, -0.0281 | 0/2 | -0.0304 [-0.0593, -0.0015] | incomplete |
| E2-g | B n=64 | 90 helps L (own delta): err(90) - err(0) | -0.0329, -0.0283 | 2/2 | -0.0306 [-0.0595, -0.0017] | incomplete |
| E2-b | B n=64 | reversed prompt beats aligned for L: err(180) - err(0) | -0.0202, -0.0349 | 2/2 | -0.0276 [-0.1206, +0.0655] | incomplete |
| E2-c | A n=64 | 180 costs about nothing for L (null-type) | -0.0024, -0.0068 | - | -0.0046 [-0.0327, +0.0236] (2 SE_seed = 0.0216) | incomplete |
| E2-a | A n=16 | zero prompt hurts L: err(zero) - err(0) | +0.1235 | - | - | incomplete |
| E2-d | A n=16 | L loses more than scratch at 90: diff-in-diff | +0.0164 | - | - | incomplete |
| E2-g | A n=16 | 90 helps L (own delta): err(90) - err(0) | +0.0163 | - | - | incomplete |
| E2-a | B n=16 | zero prompt hurts L: err(zero) - err(0) | +0.0126 | - | - | incomplete |
| E2-d | B n=16 | L loses more than scratch at 90: diff-in-diff | -0.0521 | - | - | incomplete |
| E2-g | B n=16 | 90 helps L (own delta): err(90) - err(0) | -0.0538 | - | - | incomplete |
| E2-b | B n=16 | reversed prompt beats aligned for L: err(180) - err(0) | -0.1020 | - | - | incomplete |
| E2-c | A n=16 | 180 costs about nothing for L (null-type) | +0.0099 | - | - (2 SE_seed = 0.0332) | incomplete |
| E3-a | A | L ranks better than LT, decoder probe: rho(L) - rho(LT) | -0.0032 | - | - | incomplete |
| E3-b | A | L ranks better than LT, R1 (ft seed 0): rho(L) - rho(LT) | +0.0342 | - | - | single-seed (R1 not replicated by design, 2026-10-21) |
| E3-a | B | L ranks better than LT, decoder probe: rho(L) - rho(LT) | +0.0144 | - | - | incomplete |
| E3-b | B | L ranks better than LT, R1 (ft seed 0): rho(L) - rho(LT) | +0.0034 | - | - | single-seed (R1 not replicated by design, 2026-10-21) |
