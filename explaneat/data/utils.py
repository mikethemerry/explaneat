import sklearn.preprocessing as preprocessing


def one_hot_encode(vals):
    width = max(vals)
    newVals = []
    for val in vals:
        blank = [0. for _ in range(width + 1)]
        blank[val] = 1.
        newVals.append(blank)
    return np.asarray(newVals)


def linear_scale(vals):
    scaler = preprocessing.StandardScaler()
    scaler.fit(vals)
    return scaler.transform(vals)
