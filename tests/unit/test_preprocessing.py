import pytest
from utils.preprocessing import preprocess_signal
import numpy as np

signal_emg = np.array([  0,   1,   0,   1,   1,   1,   2,  -4,   1,  11,  -6,   5,  -1,
        -3,  15,   3,   0,   1,   2,   2,   3,   0,  -1,   1,   0,  -2,
        -1,   0,   4,  -3,  -2,   7,  -6,  -1,   3,  13,  -6,   1,   1,
         3,   3,   4,   0,   0,  -1,   0,   0,  -2,  -2,  -1,  -3,   3,
         2,  -3,   0,   0,   0,   1,   0,   8,  -3,  -2,   0,   0,   4,
        -7,   2,   2,  -1,  -3,   0,  -1,   5,  -1,   0,   0,   1,  -1,
         0,   4,   0,  -2,   1,  -2,   3,  -1,   0,   1,  -1,   0,  -1,
         0,  -1,   2,  -1,  -7,  -4,  28, -13,  -4])

hilbert = np.array([ 2.73781235,  1.79413115,  1.34634937,  0.68680264,  0.6812395 ,
        0.64390741,  1.23764888,  4.78817087,  8.44697937,  9.6808556 ,
        7.48690843,  4.36850211,  7.19670644, 10.37153137,  9.83206052,
        6.46022079,  3.38226409,  2.02227579,  0.86064405,  3.01309312,
        4.15483366,  3.52612828,  2.76123581,  2.45174124,  1.97200153,
        1.95568513,  1.17136209,  2.06754182,  5.0355893 ,  6.61010334,
        6.64296224,  6.01205567,  4.00130509,  2.6791584 ,  7.32188686,
       10.03335016,  8.35090203,  3.50406507,  1.88983654,  3.84738983,
        2.75414911,  0.52929974,  1.28440249,  1.23690231,  1.94706185,
        3.33642363,  3.67038977,  2.68761882,  0.96669174,  0.85306659,
        1.85292142,  2.8040833 ,  3.80269357,  3.53454132,  1.44006153,
        1.52492539,  2.90349642,  2.32054133,  3.79740854,  6.35997282,
        6.52129971,  3.92965822,  1.51775741,  3.89340891,  5.44910555,
        5.71513269,  5.04546528,  4.21413454,  4.0752915 ,  3.42389791,
        1.12405083,  2.03302949,  4.24158158,  4.40748465,  2.96070527,
        1.2885804 ,  0.54971666,  0.89767476,  1.70763699,  2.54699494,
        2.90869075,  2.25711755,  0.68781082,  0.93781902,  1.60896462,
        1.4756526 ,  1.3496783 ,  0.83812874,  0.61056242,  2.1923619 ,
        2.76543875,  2.0248321 ,  2.97472431,  4.02239627,  2.64030546,
        8.26186745, 17.29244086, 22.75796003, 21.15617405, 13.5035943 ])

@pytest.mark.parametrize("test_input, kwargs,expected", 
                        [(signal_emg,{"SamplingRate":500,  "LF":60, "HF":240, "frequences_to_filter":[50, 100, 150, 200 ], "order_butter":4}, hilbert)])

def test_preprocess(test_input,kwargs, expected) -> None:
    assert  allclose(preprocess_signal(test_input,**kwargs), expected, atol=0.002)
