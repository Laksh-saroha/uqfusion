# Smoke, draw 941: tiered concatenation and demotion (descriptive)

Draw 941 is the smoke/calibration draw, never a scoring draw. Five Phase 3 systems, ship AP, mean over systems. Written before Amendment 4 of `docs/prereg-no-retrain-2026-10-10.md`; the scored run is on draws 945–947. `udemote` here is the unregistered arm described in the amendment (fixed-score version). `calunion` and `proben` use calibrators fitted on this same draw (leave-one-run-out), so their numbers here are in-sample.

```

== IR clean
cell                  vis       ir  shipped    union selfconf calunion   proben      lex   demote  udemote
clean/day          0.3485   0.0218   0.3566   0.3566   0.3566   0.3229   0.3290   0.3500   0.3566   0.3566
clean/night        0.2535   0.0687   0.0687   0.2895   0.2882   0.2627   0.2668   0.2725   0.1079   0.2895
fog/day            0.0116   0.0218   0.0240   0.0240   0.0216   0.0226   0.0212   0.0232   0.0240   0.0240
fog/night          0.0450   0.0687   0.0688   0.0837   0.0730   0.0871   0.0880   0.0749   0.0723   0.0837
fog_s1/day         0.0273   0.0218   0.0360   0.0360   0.0251   0.0285   0.0309   0.0365   0.0360   0.0360
fog_s1/night       0.1063   0.0687   0.0687   0.1413   0.1306   0.1306   0.1412   0.1325   0.0778   0.1413
lowlight/day       0.1712   0.0218   0.1783   0.1783   0.1635   0.1446   0.1503   0.1744   0.1783   0.1783
lowlight/night     0.2382   0.0687   0.0687   0.2705   0.2693   0.2469   0.2496   0.2548   0.1043   0.2705
glare/day          0.1584   0.0218   0.1703   0.1703   0.1505   0.1387   0.1458   0.1665   0.1703   0.1703
glare/night        0.0817   0.0687   0.0687   0.1274   0.1069   0.1230   0.1272   0.1165   0.0789   0.1274

== IR fog_s3
cell                  vis       ir  shipped    union selfconf calunion   proben      lex   demote  udemote
clean/day          0.3485   0.0033   0.3389   0.3389   0.3370   0.3464   0.3456   0.3491   0.3389   0.3399
clean/night        0.2535   0.0022   0.0070   0.2556   0.2533   0.2535   0.2525   0.2536   0.1539   0.2536
fog/day            0.0116   0.0033   0.0126   0.0126   0.0108   0.0110   0.0108   0.0129   0.0126   0.0127
fog/night          0.0450   0.0022   0.0048   0.0448   0.0451   0.0449   0.0448   0.0450   0.0167   0.0450
fog_s1/day         0.0273   0.0033   0.0286   0.0286   0.0255   0.0267   0.0257   0.0288   0.0286   0.0274
fog_s1/night       0.1063   0.0022   0.0065   0.1064   0.1062   0.1066   0.1062   0.1067   0.0477   0.1067
lowlight/day       0.1712   0.0033   0.0577   0.1684   0.1660   0.1706   0.1682   0.1719   0.0788   0.1677
lowlight/night     0.2382   0.0022   0.0069   0.2397   0.2381   0.2382   0.2373   0.2383   0.1414   0.2383
glare/day          0.1584   0.0033   0.1591   0.1591   0.1561   0.1571   0.1561   0.1600   0.1591   0.1569
glare/night        0.0817   0.0022   0.0829   0.0829   0.0818   0.0819   0.0814   0.0821   0.0829   0.0821

```

## The two udemote variants (IR clean, fog s1, fog s3)

`udemote_b` sends the flagged frames through the same union pipeline with the IR record emptied; `union_b` is that without the IR tail.

```

== IR clean
cell                   vis     union       lex   udemote udemote_b   union_b
clean/day           0.3485    0.3566    0.3500    0.3566    0.3566    0.3566
clean/night         0.2535    0.2895    0.2725    0.2895    0.2895    0.2895
fog/day             0.0116    0.0240    0.0232    0.0240    0.0240    0.0240
fog/night           0.0450    0.0837    0.0749    0.0837    0.0837    0.0837
lowlight/day        0.1712    0.1783    0.1744    0.1783    0.1783    0.1783
lowlight/night      0.2382    0.2705    0.2548    0.2705    0.2705    0.2705
glare/day           0.1584    0.1703    0.1665    0.1703    0.1703    0.1703
glare/night         0.0817    0.1274    0.1165    0.1274    0.1274    0.1274

== IR fog_s1
cell                   vis     union       lex   udemote udemote_b   union_b
clean/day           0.3485    0.3553    0.3514    0.3225    0.3255    0.3255
clean/night         0.2535    0.2849    0.2688    0.2694    0.2705    0.2668
fog/day             0.0116    0.0298    0.0288    0.0210    0.0214    0.0205
fog/night           0.0450    0.0760    0.0692    0.0692    0.0699    0.0663
lowlight/day        0.1712    0.1811    0.1765    0.1668    0.1688    0.1687
lowlight/night      0.2382    0.2660    0.2516    0.2510    0.2521    0.2489
glare/day           0.1584    0.1746    0.1700    0.1581    0.1611    0.1604
glare/night         0.0817    0.1201    0.1104    0.1100    0.1110    0.1060

== IR fog_s3
cell                   vis     union       lex   udemote udemote_b   union_b
clean/day           0.3485    0.3389    0.3491    0.3399    0.3434    0.3434
clean/night         0.2535    0.2556    0.2536    0.2536    0.2556    0.2552
fog/day             0.0116    0.0126    0.0129    0.0127    0.0122    0.0120
fog/night           0.0450    0.0448    0.0450    0.0450    0.0451    0.0451
lowlight/day        0.1712    0.1684    0.1719    0.1677    0.1710    0.1708
lowlight/night      0.2382    0.2397    0.2383    0.2383    0.2401    0.2397
glare/day           0.1584    0.1591    0.1600    0.1569    0.1609    0.1601
glare/night         0.0817    0.0829    0.0821    0.0821    0.0832    0.0827

```
