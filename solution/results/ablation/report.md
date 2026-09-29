| variant | frames | TP | FP | FN | TN | precision | recall | false-alarm events | obstacle detected (first frame / distance) | proc ms median (max of bags) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| full | 1246 | 24 | 0 | 0 | 1079 | 1.000 | 1.000 | 0 | obstacle_1: yes (16 / 55.662 m) | 72 |
| no_temporal | 1246 | 24 | 7 | 0 | 1072 | 0.774 | 1.000 | 4 | obstacle_1: yes (16 / 55.662 m) | 82 |
| box_only | 1246 | 24 | 111 | 0 | 968 | 0.178 | 1.000 | 11 | obstacle_1: yes (16 / 55.662 m) | 98 |
| fixed_thresholds | 1246 | 24 | 54 | 0 | 1025 | 0.308 | 1.000 | 12 | obstacle_1: yes (16 / 55.676 m) | 83 |
| straight_corridor | 1246 | 0 | 28 | 24 | 1051 | 0.000 | 0.000 | 8 | obstacle_1: NO (- / - m) | 89 |
| no_voxel | 1246 | 24 | 2 | 0 | 1077 | 0.923 | 1.000 | 2 | obstacle_1: yes (16 / 55.67 m) | 78 |

### full
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 24 | 0 | 0 | 0 | 77 | - | 71.8 / 86.2 / 88.1 |
| doubleT_platform | 173 | 0 | 0 | 0 | 173 | 0 | - | 38.7 / 67.3 / 73.5 |
| roundT_doubleT | 126 | 0 | 0 | 0 | 126 | 0 | - | 45.1 / 75.5 / 87.5 |
| roundT_pressureGate_roundT | 134 | 0 | 0 | 0 | 134 | 0 | - | 39.8 / 61.5 / 71.6 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 0 | 0 | 273 | 0 | - | 32.7 / 65.6 / 69.0 |
| squareT_platform_squareT_switch | 439 | 0 | 0 | 0 | 373 | 66 | - | 29.2 / 61.1 / 165.7 |

### no_temporal
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 24 | 0 | 0 | 0 | 77 | - | 81.8 / 89.9 / 102.3 |
| doubleT_platform | 173 | 0 | 4 | 0 | 169 | 0 | [68, 70] t=6.8-7.0s [63.04, 95.304]; [122, 126] t=12.2-12.6s [102.964, 106.484] | 40.0 / 68.3 / 72.9 |
| roundT_doubleT | 126 | 0 | 1 | 0 | 125 | 0 | [40, 40] t=4.0-4.0s [114.468, 114.468] | 45.3 / 77.5 / 83.6 |
| roundT_pressureGate_roundT | 134 | 0 | 0 | 0 | 134 | 0 | - | 50.0 / 74.9 / 94.4 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 2 | 0 | 271 | 0 | [296, 298] t=30.6-30.8s [108.072, 111.88] | 41.9 / 82.6 / 190.8 |
| squareT_platform_squareT_switch | 439 | 0 | 0 | 0 | 373 | 66 | - | 37.3 / 85.5 / 209.4 |

### box_only
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 24 | 0 | 0 | 0 | 77 | - | 98.1 / 105.7 / 109.5 |
| doubleT_platform | 173 | 0 | 30 | 0 | 143 | 0 | [62, 64] t=6.2-6.4s [3.691, 3.874]; [72, 76] t=7.2-7.6s [22.608, 45.632]; [84, 138] t=8.4-13.8s [3.348, 112.668] | 47.0 / 90.3 / 104.3 |
| roundT_doubleT | 126 | 0 | 14 | 0 | 112 | 0 | [52, 54] t=5.2-5.4s [124.656, 131.724]; [114, 126] t=11.4-12.6s [37.058, 62.691]; [232, 244] t=23.2-24.4s [90.528, 116.888] | 54.9 / 92.9 / 100.6 |
| roundT_pressureGate_roundT | 134 | 0 | 2 | 0 | 132 | 0 | [52, 54] t=5.2-5.4s [121.956, 125.0] | 41.7 / 62.6 / 81.7 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 1 | 0 | 272 | 0 | [292, 292] t=30.2-30.2s [145.296, 145.296] | 32.8 / 67.5 / 80.7 |
| squareT_platform_squareT_switch | 439 | 0 | 64 | 0 | 309 | 66 | [96, 174] t=9.6-18.0s [3.718, 27.356]; [182, 228] t=18.8-23.4s [3.497, 4.03]; [236, 236] t=24.2-24.2s [3.366, 3.366] | 28.9 / 74.9 / 183.6 |

### fixed_thresholds
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 24 | 0 | 0 | 0 | 77 | - | 83.3 / 88.1 / 91.2 |
| doubleT_platform | 173 | 0 | 24 | 0 | 149 | 0 | [48, 48] t=4.8-4.8s [77.908, 77.908]; [58, 62] t=5.8-6.2s [73.412, 78.712]; [70, 76] t=7.0-7.6s [45.772, 57.888]; [82, 88] t=8.2-8.8s [57.1, 71.044]; [96, 110] t=9.6-11.0s [75.6, 112.632]; [124, 138] t=12.4-13.8s [45.916, 57.528] | 65.1 / 84.6 / 97.0 |
| roundT_doubleT | 126 | 0 | 7 | 0 | 119 | 0 | [112, 124] t=11.2-12.4s [88.996, 99.113] | 66.6 / 84.7 / 96.8 |
| roundT_pressureGate_roundT | 134 | 0 | 1 | 0 | 133 | 0 | [170, 170] t=17.0-17.0s [104.257, 104.257] | 68.5 / 77.7 / 92.2 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 1 | 0 | 272 | 0 | [300, 300] t=31.0-31.0s [115.952, 115.952] | 53.5 / 68.1 / 71.4 |
| squareT_platform_squareT_switch | 439 | 0 | 21 | 0 | 352 | 66 | [98, 124] t=9.8-12.4s [20.348, 26.112]; [132, 144] t=13.2-15.0s [21.228, 27.732]; [154, 154] t=16.0-16.0s [25.116, 25.116] | 45.1 / 83.6 / 226.0 |

### straight_corridor
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 0 | 0 | 24 | 0 | 77 | - | 85.2 / 89.7 / 93.8 |
| doubleT_platform | 173 | 0 | 1 | 0 | 172 | 0 | [210, 210] t=21.0-21.0s [65.996, 65.996] | 62.4 / 84.7 / 94.5 |
| roundT_doubleT | 126 | 0 | 12 | 0 | 114 | 0 | [4, 8] t=0.4-0.8s [78.884, 96.204]; [98, 116] t=9.8-11.6s [22.656, 41.684] | 66.4 / 81.3 / 97.6 |
| roundT_pressureGate_roundT | 134 | 0 | 10 | 0 | 124 | 0 | [106, 112] t=10.6-11.2s [31.08, 52.384]; [120, 122] t=12.0-12.2s [29.424, 32.376]; [164, 172] t=16.4-17.2s [24.872, 33.448] | 89.0 / 119.4 / 128.8 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 5 | 0 | 268 | 0 | [330, 336] t=34.0-34.6s [83.448, 91.98]; [346, 348] t=35.6-35.8s [51.308, 54.049] | 89.3 / 110.4 / 124.7 |
| squareT_platform_squareT_switch | 439 | 0 | 0 | 0 | 373 | 66 | - | 42.7 / 108.7 / 154.9 |

### no_voxel
| bag | frames | TP | FP | FN | TN | uncertain | false alarms (frames, t, distance) | proc ms median / p95 / max |
|---|---:|---:|---:|---:|---:|---:|---|---|
| doubleT_obstacle | 101 | 24 | 0 | 0 | 0 | 77 | - | 78.4 / 84.2 / 88.1 |
| doubleT_platform | 173 | 0 | 2 | 0 | 171 | 0 | [72, 72] t=7.2-7.2s [60.54, 60.54]; [124, 124] t=12.4-12.4s [57.528, 57.528] | 53.2 / 2235.3 / 3367.5 |
| roundT_doubleT | 126 | 0 | 0 | 0 | 126 | 0 | - | 43.7 / 73.4 / 544.8 |
| roundT_pressureGate_roundT | 134 | 0 | 0 | 0 | 134 | 0 | - | 39.0 / 63.0 / 72.7 |
| roundT_squareT_pressureGate_squareT | 273 | 0 | 0 | 0 | 273 | 0 | - | 37.2 / 63.6 / 67.2 |
| squareT_platform_squareT_switch | 439 | 0 | 0 | 0 | 373 | 66 | - | 26.9 / 2567.7 / 14040.7 |

