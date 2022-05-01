import re
import numpy as np
import torch as pt

# https://stackoverflow.com/questions/35750639/how-can-a-string-representation-of-a-numpy-array-be-converted-to-a-numpy-array
#credit to the user normanius
def string_to_numpy(text, dtype=None):
    """
    Convert text into 1D or 2D arrays using np.matrix().
    The result is returned as an np.ndarray.
    """
    import re
    text = text.strip()
    # Using a regexp, decide whether the array is flat or not.
    # The following matches either: "[1 2 3]" or "1 2 3"
    is_flat = bool(re.match(r"^(\[[^\[].+[^\]]\]|[^\[].+[^\]])$",
                            text, flags=re.S))
    # Replace newline characters with semicolons.
    text = text.replace("]\n", "];")
    # Prepare the result.
    result = np.asarray(np.matrix(text, dtype=dtype))
    return result.flatten() if is_flat else result


def sum_rows(t):
    """
    Return the sums of a tensor over the first dimension.
    """
    sums = list(map(sum, t)) 
    return sums

def max_val_pos(t):
    """
    Get the position of the max value in a 2D tensor
    The result is a pair of coordinates
    """
    vals, max_row_in_cols = t.max(0)

    col_idx = int(vals.argmax(0))
    row_idx = int(max_row_in_cols[col_idx])
    
    
    return row_idx, col_idx


def k_max_val_pos(t, k = 1):
    """
    Get  positions of k maximal values in a 2D tensor.
    The result is a list of k positions (pairs).
    """
    h, w  = t.shape
    
    #topk returns values and indices, [1] -> indices
    flat_ids = pt.topk(t.flatten(),k)[1]
    
    matrix_ids = list(map(lambda x: (int(x//w),int(x%w)),flat_ids))
     
    return matrix_ids


output = """[[8.0000e+00 1.2463e+04 1.2840e+03 3.0200e+02 4.0800e+02 2.2400e+02
  1.8720e+03 1.5410e+03 1.2700e+02 2.1310e+03 8.9700e+02 0.0000e+00]
 [1.1900e+02 6.0000e+00 2.5010e+03 4.4100e+02 2.6860e+03 1.4400e+02
  2.3600e+02 1.3300e+02 5.8400e+02 2.8300e+02 2.3400e+02 6.8900e+02]
 [1.9080e+03 2.8970e+03 3.1900e+02 1.5440e+03 1.9400e+02 2.6700e+02
  2.1000e+01 4.0000e+00 2.9000e+01 2.3000e+02 2.9100e+02 4.0000e+00]
 [4.0000e+00 2.0000e+00 0.0000e+00 0.0000e+00 2.0000e+00 1.0000e+00
  0.0000e+00 2.4000e+01 0.0000e+00 3.8000e+02 0.0000e+00 4.2800e+03]
 [1.4000e+01 0.0000e+00 2.6700e+02 3.0000e+00 0.0000e+00 1.1700e+02
  1.6000e+01 0.0000e+00 0.0000e+00 8.4000e+02 0.0000e+00 0.0000e+00]
 [0.0000e+00 3.3350e+03 1.4000e+01 0.0000e+00 1.0000e+01 1.0000e+00
  4.1000e+01 7.0000e+00 6.0000e+00 1.9700e+02 1.4000e+01 0.0000e+00]
 [0.0000e+00 1.2690e+03 1.9100e+02 9.0000e+00 5.0000e+00 1.1000e+02
  0.0000e+00 6.0000e+00 4.0000e+00 6.7500e+02 5.7000e+01 0.0000e+00]
 [1.6230e+03 4.2000e+01 2.3000e+01 1.0000e+00 4.5000e+01 0.0000e+00
  3.0000e+00 0.0000e+00 2.0000e+00 0.0000e+00 0.0000e+00 3.0000e+00]
 [1.0000e+02 1.0000e+00 8.6000e+01 8.0000e+00 4.0000e+00 3.8000e+01
  0.0000e+00 0.0000e+00 3.4000e+01 3.3000e+01 0.0000e+00 1.0000e+00]
 [2.0000e+00 1.7000e+01 1.7500e+02 2.0000e+00 3.2000e+01 0.0000e+00
  2.0000e+00 1.0000e+00 2.2000e+01 1.0000e+00 0.0000e+00 1.5000e+01]
 [0.0000e+00 1.1000e+01 3.7000e+01 3.5000e+01 7.0000e+00 0.0000e+00
  4.9000e+02 3.0700e+02 0.0000e+00 0.0000e+00 1.0000e+00 0.0000e+00]
 [0.0000e+00 0.0000e+00 3.5000e+02 4.0100e+02 3.0000e+00 0.0000e+00
  0.0000e+00 0.0000e+00 0.0000e+00 2.0800e+02 0.0000e+00 4.7100e+02]]

[[1.49367987e-04 2.32696652e-01 2.39735619e-02 5.63864150e-03
  7.61776732e-03 4.18230363e-03 3.49521089e-02 2.87720084e-02
  2.37121679e-03 3.97878975e-02 1.67478855e-02 0.00000000e+00]
 [2.22184880e-03 1.12025990e-04 4.66961668e-02 8.23391027e-03
  5.01503015e-02 2.68862376e-03 4.40635561e-03 2.48324278e-03
  1.09038630e-02 5.28389253e-03 4.36901361e-03 1.28643179e-02]
 [3.56242648e-02 5.40898822e-02 5.95604847e-03 2.88280214e-02
  3.62217368e-03 4.98515656e-03 3.92090965e-04 7.46839934e-05
  5.41458952e-04 4.29432962e-03 5.43326052e-03 7.46839934e-05]
 [7.46839934e-05 3.73419967e-05 0.00000000e+00 0.00000000e+00
  3.73419967e-05 1.86709983e-05 0.00000000e+00 4.48103960e-04
  0.00000000e+00 7.09497937e-03 0.00000000e+00 7.99118729e-02]
 [2.61393977e-04 0.00000000e+00 4.98515656e-03 5.60129950e-05
  0.00000000e+00 2.18450681e-03 2.98735973e-04 0.00000000e+00
  0.00000000e+00 1.56836386e-02 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 6.22677795e-02 2.61393977e-04 0.00000000e+00
  1.86709983e-04 1.86709983e-05 7.65510932e-04 1.30696988e-04
  1.12025990e-04 3.67818667e-03 2.61393977e-04 0.00000000e+00]
 [0.00000000e+00 2.36934969e-02 3.56616068e-03 1.68038985e-04
  9.33549917e-05 2.05380982e-03 0.00000000e+00 1.12025990e-04
  7.46839934e-05 1.26029239e-02 1.06424691e-03 0.00000000e+00]
 [3.03030303e-02 7.84181930e-04 4.29432962e-04 1.86709983e-05
  8.40194925e-04 0.00000000e+00 5.60129950e-05 0.00000000e+00
  3.73419967e-05 0.00000000e+00 0.00000000e+00 5.60129950e-05]
 [1.86709983e-03 1.86709983e-05 1.60570586e-03 1.49367987e-04
  7.46839934e-05 7.09497937e-04 0.00000000e+00 0.00000000e+00
  6.34813944e-04 6.16142945e-04 0.00000000e+00 1.86709983e-05]
 [3.73419967e-05 3.17406972e-04 3.26742471e-03 3.73419967e-05
  5.97471947e-04 0.00000000e+00 3.73419967e-05 1.86709983e-05
  4.10761963e-04 1.86709983e-05 0.00000000e+00 2.80064975e-04]
 [0.00000000e+00 2.05380982e-04 6.90826939e-04 6.53484942e-04
  1.30696988e-04 0.00000000e+00 9.14878919e-03 5.73199649e-03
  0.00000000e+00 0.00000000e+00 1.86709983e-05 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 6.53484942e-03 7.48707033e-03
  5.60129950e-05 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 3.88356765e-03 0.00000000e+00 8.79404022e-03]]

[[1.0360e+03 2.6551e+04 4.2750e+03 9.5900e+02 2.0580e+03 3.5700e+03
  6.9110e+03 4.8650e+03 2.1070e+03 6.9070e+03 2.6120e+03 6.0810e+03]
 [1.2730e+03 5.6200e+02 6.1250e+03 1.3200e+03 9.9580e+03 9.8200e+02
  2.4480e+03 1.0180e+03 1.9510e+03 1.3170e+03 1.2080e+03 2.2330e+03]
 [4.0460e+03 7.7930e+03 9.0800e+02 3.7910e+03 7.0800e+02 5.7300e+02
  9.0000e+01 3.2000e+01 1.5700e+02 8.5700e+02 7.7600e+02 7.8500e+02]
 [1.6000e+01 1.5000e+01 9.0000e+00 1.3800e+02 2.8000e+01 1.8240e+03
  0.0000e+00 2.2600e+02 6.8800e+02 1.0530e+03 1.5700e+02 1.5983e+04]
 [6.8000e+01 1.0000e+00 2.0280e+03 7.9000e+01 4.0000e+00 7.5900e+02
  5.6000e+01 0.0000e+00 1.1000e+01 2.3060e+03 1.4290e+03 0.0000e+00]
 [1.0000e+00 1.1743e+04 1.1200e+02 0.0000e+00 3.1000e+01 4.0000e+00
  2.9100e+02 5.7000e+01 1.2000e+02 6.9900e+02 6.3000e+01 0.0000e+00]
 [3.0000e+00 7.0380e+03 3.9800e+02 5.9500e+02 4.3900e+02 5.9630e+03
  2.0100e+02 2.1000e+02 3.2000e+01 2.4580e+03 3.0200e+02 9.6000e+01]
 [1.8133e+04 1.9100e+02 7.7000e+01 1.8000e+01 3.6200e+02 1.7000e+01
  2.0250e+03 4.3000e+01 2.2000e+01 2.0000e+00 4.1400e+02 2.1000e+01]
 [1.3940e+03 3.7000e+01 2.2110e+03 1.4200e+02 2.4000e+01 1.5800e+02
  9.8410e+03 1.9110e+03 4.0640e+03 2.1700e+02 1.8500e+02 1.2830e+03]
 [4.0000e+01 3.7300e+02 3.0830e+03 1.7900e+02 1.8000e+02 5.4400e+02
  1.4740e+03 9.0000e+00 3.3640e+03 1.7640e+03 4.6240e+03 7.4000e+01]
 [2.0560e+03 1.9870e+03 5.2300e+02 1.0540e+03 3.0000e+01 0.0000e+00
  4.1750e+03 2.5880e+03 2.9300e+02 1.0000e+00 1.8000e+01 1.1100e+02]
 [0.0000e+00 0.0000e+00 1.2240e+03 1.4960e+03 9.0000e+00 0.0000e+00
  0.0000e+00 0.0000e+00 0.0000e+00 1.7570e+03 0.0000e+00 2.4740e+03]]

[[4.06476951e-03 1.04173451e-01 1.67730595e-02 3.76265827e-03
  8.07460971e-03 1.40069760e-02 2.71154653e-02 1.90879379e-02
  8.26686232e-03 2.70997713e-02 1.02482413e-02 2.38589415e-02]
 [4.99464439e-03 2.20501975e-03 2.40315765e-02 5.17904996e-03
  3.90704390e-02 3.85289929e-03 9.60478356e-03 3.99414610e-03
  7.65479278e-03 5.16727939e-03 4.73961542e-03 8.76122618e-03]
 [1.58745728e-02 3.05760124e-02 3.56255861e-03 1.48740745e-02
  2.77785407e-03 2.24817850e-03 3.53117043e-04 1.25552726e-04
  6.15993063e-04 3.36245895e-03 3.04465361e-03 3.07996532e-03]
 [6.27763631e-05 5.88528404e-05 3.53117043e-05 5.41446132e-04
  1.09858635e-04 7.15650540e-03 0.00000000e+00 8.86716129e-04
  2.69938361e-03 4.13146940e-03 6.15993063e-04 6.27096632e-02]
 [2.66799543e-04 3.92352270e-06 7.95690403e-03 3.09958293e-04
  1.56940908e-05 2.97795373e-03 2.19717271e-04 0.00000000e+00
  4.31587497e-05 9.04764334e-03 5.60671393e-03 0.00000000e+00]
 [3.92352270e-06 4.60739270e-02 4.39434542e-04 0.00000000e+00
  1.21629204e-04 1.56940908e-05 1.14174510e-03 2.23640794e-04
  4.70822723e-04 2.74254236e-03 2.47181930e-04 0.00000000e+00]
 [1.17705681e-05 2.76137527e-02 1.56156203e-03 2.33449600e-03
  1.72242646e-03 2.33959658e-02 7.88628062e-04 8.23939766e-04
  1.25552726e-04 9.64401879e-03 1.18490385e-03 3.76658179e-04]
 [7.11452370e-02 7.49392835e-04 3.02111248e-04 7.06234085e-05
  1.42031522e-03 6.66998858e-05 7.94513346e-03 1.68711476e-04
  8.63174993e-05 7.84704539e-06 1.62433840e-03 8.23939766e-05]
 [5.46939064e-03 1.45170340e-04 8.67490868e-03 5.57140223e-04
  9.41645447e-05 6.19916586e-04 3.86113868e-02 7.49785187e-03
  1.59451962e-02 8.51404425e-04 7.25851699e-04 5.03387962e-03]
 [1.56940908e-04 1.46347397e-03 1.20962205e-02 7.02310563e-04
  7.06234085e-04 2.13439635e-03 5.78327245e-03 3.53117043e-05
  1.31987303e-02 6.92109404e-03 1.81423689e-02 2.90340679e-04]
 [8.06676266e-03 7.79603960e-03 2.05200237e-03 4.13539292e-03
  1.17705681e-04 0.00000000e+00 1.63807073e-02 1.01540767e-02
  1.14959215e-03 3.92352270e-06 7.06234085e-05 4.35511019e-04]
 [0.00000000e+00 0.00000000e+00 4.80239178e-03 5.86958995e-03
  3.53117043e-05 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 6.89362938e-03 0.00000000e+00 9.70679515e-03]]

[[0.000e+00 4.530e+02 1.778e+03 7.320e+02 3.760e+02 4.680e+02 1.923e+03
  3.100e+01 8.000e+00 1.305e+03 1.400e+01 0.000e+00]
 [0.000e+00 7.000e+00 1.210e+02 1.407e+03 2.040e+02 4.000e+00 3.000e+00
  9.000e+00 0.000e+00 0.000e+00 3.300e+01 3.260e+03]
 [2.933e+03 5.350e+02 4.330e+02 4.731e+03 8.540e+02 1.560e+02 3.000e+00
  0.000e+00 5.600e+01 1.910e+02 1.689e+03 4.000e+00]
 [8.000e+00 0.000e+00 0.000e+00 0.000e+00 2.000e+01 2.000e+00 0.000e+00
  5.800e+01 0.000e+00 5.160e+02 0.000e+00 6.920e+02]
 [3.500e+01 0.000e+00 1.170e+02 2.300e+01 0.000e+00 3.180e+02 4.000e+00
  0.000e+00 2.000e+00 9.780e+02 0.000e+00 0.000e+00]
 [0.000e+00 6.070e+02 1.800e+01 0.000e+00 1.230e+02 2.000e+00 2.000e+02
  1.100e+01 2.400e+01 5.000e+01 1.000e+01 0.000e+00]
 [1.000e+00 4.300e+02 8.000e+00 1.300e+01 1.040e+02 3.000e+00 0.000e+00
  1.600e+01 1.600e+01 3.346e+03 1.890e+02 0.000e+00]
 [1.400e+02 5.000e+01 1.000e+01 1.000e+00 3.200e+02 0.000e+00 1.000e+00
  0.000e+00 9.000e+00 0.000e+00 0.000e+00 0.000e+00]
 [5.400e+01 7.000e+00 0.000e+00 1.000e+00 5.000e+00 2.400e+01 0.000e+00
  0.000e+00 1.000e+00 2.280e+02 4.000e+00 3.000e+00]
 [4.380e+02 6.000e+00 1.300e+01 0.000e+00 2.400e+01 0.000e+00 3.000e+00
  0.000e+00 1.100e+01 0.000e+00 0.000e+00 5.700e+01]
 [0.000e+00 8.000e+00 1.100e+01 1.000e+01 0.000e+00 0.000e+00 1.380e+02
  1.910e+02 0.000e+00 0.000e+00 1.000e+00 0.000e+00]
 [0.000e+00 0.000e+00 1.810e+02 2.460e+02 1.000e+00 0.000e+00 0.000e+00
  0.000e+00 0.000e+00 2.890e+02 0.000e+00 2.400e+02]]

[[0.00000000e+00 1.31720508e-02 5.16995726e-02 2.12846384e-02
  1.09330930e-02 1.36082115e-02 5.59157919e-02 9.01398622e-04
  2.32618999e-04 3.79459742e-02 4.07083249e-04 0.00000000e+00]
 [0.00000000e+00 2.03541624e-04 3.51836236e-03 4.09118665e-02
  5.93178448e-03 1.16309500e-04 8.72321247e-05 2.61696374e-04
  0.00000000e+00 0.00000000e+00 9.59553372e-04 9.47922422e-02]
 [8.52839406e-02 1.55563956e-02 1.25905033e-02 1.37565061e-01
  2.48320782e-02 4.53607048e-03 8.72321247e-05 0.00000000e+00
  1.62833299e-03 5.55377860e-03 4.91116862e-02 1.16309500e-04]
 [2.32618999e-04 0.00000000e+00 0.00000000e+00 0.00000000e+00
  5.81547498e-04 5.81547498e-05 0.00000000e+00 1.68648774e-03
  0.00000000e+00 1.50039254e-02 0.00000000e+00 2.01215434e-02]
 [1.01770812e-03 0.00000000e+00 3.40205286e-03 6.68779623e-04
  0.00000000e+00 9.24660522e-03 1.16309500e-04 0.00000000e+00
  5.81547498e-05 2.84376726e-02 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 1.76499666e-02 5.23392748e-04 0.00000000e+00
  3.57651711e-03 5.81547498e-05 5.81547498e-03 3.19851124e-04
  6.97856997e-04 1.45386874e-03 2.90773749e-04 0.00000000e+00]
 [2.90773749e-05 1.25032712e-02 2.32618999e-04 3.78005874e-04
  3.02404699e-03 8.72321247e-05 0.00000000e+00 4.65237998e-04
  4.65237998e-04 9.72928964e-02 5.49562386e-03 0.00000000e+00]
 [4.07083249e-03 1.45386874e-03 2.90773749e-04 2.90773749e-05
  9.30475997e-03 0.00000000e+00 2.90773749e-05 0.00000000e+00
  2.61696374e-04 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [1.57017824e-03 2.03541624e-04 0.00000000e+00 2.90773749e-05
  1.45386874e-04 6.97856997e-04 0.00000000e+00 0.00000000e+00
  2.90773749e-05 6.62964148e-03 1.16309500e-04 8.72321247e-05]
 [1.27358902e-02 1.74464249e-04 3.78005874e-04 0.00000000e+00
  6.97856997e-04 0.00000000e+00 8.72321247e-05 0.00000000e+00
  3.19851124e-04 0.00000000e+00 0.00000000e+00 1.65741037e-03]
 [0.00000000e+00 2.32618999e-04 3.19851124e-04 2.90773749e-04
  0.00000000e+00 0.00000000e+00 4.01267774e-03 5.55377860e-03
  0.00000000e+00 0.00000000e+00 2.90773749e-05 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 5.26300486e-03 7.15303422e-03
  2.90773749e-05 0.00000000e+00 0.00000000e+00 0.00000000e+00
  0.00000000e+00 8.40336134e-03 0.00000000e+00 6.97856997e-03]]

[[3.0000e+02 1.5750e+03 6.0440e+03 2.9090e+03 2.1820e+03 6.4470e+03
  7.0750e+03 2.2100e+02 2.7370e+03 4.2310e+03 6.2000e+01 5.9900e+03]
 [4.5000e+01 1.4360e+03 4.1000e+02 5.0690e+03 7.6500e+02 5.5000e+01
  1.1000e+02 1.0300e+02 0.0000e+00 1.0000e+00 1.0600e+02 1.3090e+04]
 [5.3360e+03 1.0940e+03 1.0910e+03 1.6012e+04 2.6640e+03 3.7800e+02
  2.8000e+01 1.5000e+01 1.3200e+02 5.4400e+02 3.5010e+03 5.4600e+02]
 [3.2000e+01 5.0000e+00 1.0000e+00 1.7500e+02 7.1200e+02 1.3300e+02
  0.0000e+00 4.2600e+02 4.2500e+02 1.3700e+03 4.0000e+00 1.9830e+03]
 [8.0000e+01 1.0000e+00 6.3500e+02 2.0900e+02 2.8000e+01 2.5970e+03
  1.3000e+01 0.0000e+00 1.6500e+02 3.9940e+03 4.0000e+02 0.0000e+00]
 [4.0000e+00 1.5980e+03 9.5000e+01 0.0000e+00 3.1400e+02 3.0000e+00
  1.7340e+03 4.2000e+01 4.7700e+02 2.1000e+02 1.5000e+01 2.0000e+00]
 [7.0000e+00 9.4200e+02 3.1000e+01 9.9000e+02 8.0030e+03 1.0900e+02
  3.0000e+00 1.5260e+03 2.6900e+02 1.2910e+04 1.0840e+03 1.9100e+02]
 [9.9800e+02 1.3400e+02 1.2100e+02 8.2000e+01 2.2220e+03 1.1000e+02
  4.4000e+01 1.2000e+01 7.2000e+01 0.0000e+00 3.2000e+01 6.0000e+00]
 [3.3700e+02 3.5900e+02 0.0000e+00 7.4000e+01 1.9000e+01 1.2200e+02
  2.2800e+02 7.4700e+02 3.3300e+02 1.1230e+03 1.3330e+03 2.6000e+02]
 [5.2100e+03 2.9000e+02 3.7100e+02 7.2000e+01 1.1000e+02 4.1700e+02
  5.1800e+02 5.0000e+00 3.1450e+03 2.7890e+03 1.0760e+03 2.1900e+02]
 [2.8030e+03 6.7500e+02 4.4400e+02 5.5000e+02 2.0000e+00 0.0000e+00
  1.8340e+03 1.5250e+03 1.3200e+02 0.0000e+00 4.0000e+00 9.0000e+00]
 [0.0000e+00 1.0000e+00 7.0900e+02 1.1680e+03 1.0000e+01 0.0000e+00
  0.0000e+00 0.0000e+00 3.0000e+00 2.4930e+03 0.0000e+00 1.0300e+03]]

[[1.73779057e-03 9.12340051e-03 3.50106874e-02 1.68507759e-02
  1.26395301e-02 3.73451194e-02 4.09828943e-02 1.28017239e-03
  1.58544427e-02 2.45086397e-02 3.59143385e-04 3.46978851e-02]
 [2.60668586e-04 8.31822421e-03 2.37498045e-03 2.93628680e-02
  4.43136596e-03 3.18594938e-04 6.37189877e-04 5.96641430e-04
  0.00000000e+00 5.79263524e-06 6.14019336e-04 7.58255953e-02]
 [3.09095017e-02 6.33714296e-03 6.31976505e-03 9.27516755e-02
  1.54315803e-02 2.18961612e-03 1.62193787e-04 8.68895287e-05
  7.64627852e-04 3.15119357e-03 2.02800160e-02 3.16277884e-03]
 [1.85364328e-04 2.89631762e-05 5.79263524e-06 1.01371117e-03
  4.12435629e-03 7.70420487e-04 0.00000000e+00 2.46766261e-03
  2.46186998e-03 7.93591028e-03 2.31705410e-05 1.14867957e-02]
 [4.63410819e-04 5.79263524e-06 3.67832338e-03 1.21066077e-03
  1.62193787e-04 1.50434737e-02 7.53042582e-05 0.00000000e+00
  9.55784815e-04 2.31357852e-02 2.31705410e-03 0.00000000e+00]
 [2.31705410e-05 9.25663112e-03 5.50300348e-04 0.00000000e+00
  1.81888747e-03 1.73779057e-05 1.00444295e-02 2.43290680e-04
  2.76308701e-03 1.21645340e-03 8.68895287e-05 1.15852705e-05]
 [4.05484467e-05 5.45666240e-03 1.79571693e-04 5.73470889e-03
  4.63584599e-02 6.31397242e-04 1.73779057e-05 8.83956138e-03
  1.55821888e-03 7.47829210e-02 6.27921660e-03 1.10639333e-03]
 [5.78104997e-03 7.76213123e-04 7.00908864e-04 4.74996090e-04
  1.28712355e-02 6.37189877e-04 2.54875951e-04 6.95116229e-05
  4.17069738e-04 0.00000000e+00 1.85364328e-04 3.47558115e-05]
 [1.95211808e-03 2.07955605e-03 0.00000000e+00 4.28655008e-04
  1.10060070e-04 7.06701500e-04 1.32072084e-03 4.32709853e-03
  1.92894754e-03 6.50512938e-03 7.72158278e-03 1.50608516e-03]
 [3.01796296e-02 1.67986422e-03 2.14906768e-03 4.17069738e-04
  6.37189877e-04 2.41552890e-03 3.00058506e-03 2.89631762e-05
  1.82178378e-02 1.61556597e-02 6.23287552e-03 1.26858712e-03]
 [1.62367566e-02 3.91002879e-03 2.57193005e-03 3.18594938e-03
  1.15852705e-05 0.00000000e+00 1.06236930e-02 8.83376875e-03
  7.64627852e-04 0.00000000e+00 2.31705410e-05 5.21337172e-05]
 [0.00000000e+00 5.79263524e-06 4.10697839e-03 6.76579796e-03
  5.79263524e-05 0.00000000e+00 0.00000000e+00 0.00000000e+00
  1.73779057e-05 1.44410397e-02 0.00000000e+00 5.96641430e-03]]"""

results = list(map(lambda x: string_to_numpy(x, float), output.split("\n\n")))

def show_res():
    for i, m in enumerate(results):
        l = sum_rows(m)
        if (i%2 == 0):
            print(list(map(int, l)))
        else:
            print(list(map(lambda x: f"{x*100:.2f}", l)))

#if __name__ == "__main__": 
"""
[21257, 8056, 7708, 4693, 1257, 3625, 2326, 1742, 305, 269, 888, 1433]
['39.69', '15.04', '14.39', '8.76', '2.35', '6.77', '4.34', '3.25', '0.57', '0.50', '1.66', '2.68']
[67932, 30395, 20516, 20137, 6741, 13121, 17735, 21325, 21467, 15708, 12836, 6960]
['26.65', '11.93', '8.05', '7.90', '2.64', '5.15', '6.96', '8.37', '8.42', '6.16', '5.04', '2.73']
[7088, 5048, 11585, 1296, 1477, 1045, 4126, 531, 327, 552, 359, 957]
['20.61', '14.68', '33.69', '3.77', '4.29', '3.04', '12.00', '1.54', '0.95', '1.61', '1.04', '2.78']
[39773, 21190, 31341, 5266, 8122, 4494, 26065, 3833, 4935, 14222, 7978, 5414]
['23.04', '12.27', '18.15', '3.05', '4.70', '2.60', '15.10', '2.22', '2.86', '8.24', '4.62', '3.14']

layer & sv & sv_ratio & sv3 & sv3_ratio & vo & vo_ratio & vo & vo_ratio \\ 
 1 & 21257 & 39.69 & 67932 & 26.65 & 7088 & 20.61 & 39773 & 23.04 \\
 2 & 8056 & 15.04 & 30395 & 11.93 & 5048 & 14.68 & 21190 & 12.27\\
 3 & 7708 & 14.39 & 20516 & 8.05 & 11585 & 33.69 & 31341 & 18.15\\
 4 & 4693 & 8.76 & 20137 & 7.90 & 1296 & 3.77 & 5266 & 3.05\\
 5 & 1257 & 2.35 & 6741 & 2.64 & 1477 & 4.29 & 8122 & 4.70\\
 6 & 3625 & 6.77 & 13121 & 5.15 & 1045 & 3.04 & 4494 & 2.60\\
 7 & 2326 & 4.34 & 17735 & 6.96 & 4126 & 12.00 & 26065 & 15.10\\
 8 & 1742 & 3.25 & 21325 & 8.37 & 531 & 1.54 & 3833 & 2.22\\
 9 & 305 & 0.57 & 21467 & 8.42 & 327 & 0.95 & 4935 & 2.86\\
 10 & 269 & 0.50 & 15708 & 6.16 & 552 & 1.61 & 14222 & 8.24\\
 11 & 888 & 1.66 & 12836 & 5.04 & 359 & 1.04 & 7978 & 4.62\\
 12 & 1433 & 2.68 & 6960 & 2.73 & 957 & 2.78 & 5414 & 3.14\\
 """
