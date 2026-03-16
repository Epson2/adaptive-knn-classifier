import numpy as np

def train_model(X, y):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xs = (X - mean) / std

		#extract positive class cluster
    pos = Xs[y == 1]

    #covariance for anisotropic distance
    cov = np.cov(pos.T) + 1e-6 * np.eye(2)
    cov_inv = np.linalg.inv(cov)

    return {
        "X_train": Xs,
        "y_train": y,
        "pos": pos,
        "mean": mean,
        "std": std,
        "cov_inv": cov_inv
    }


def run_model(model, X_test):
    mean = model["mean"]
    std = model["std"]
    X_train = model["X_train"]
    y_train = model["y_train"]
    pos = model["pos"]
    cov_inv = model["cov_inv"]

		#test data
    Xs = (X_test - mean) / std
    preds = []

    for x in Xs:
        #mahalanobis distance to ALL
        diff = X_train - x
        d_all = np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))

        #mahalanobis distance to positive cluster only
        diff_pos = pos - x
        d_pos = np.sqrt(np.sum((diff_pos @ cov_inv) * diff_pos, axis=1))

        
        tight_R = np.mean(np.sort(d_pos)[:12]) * 1.05   
        loose_R = np.mean(np.sort(d_pos)[:30]) * 1.75   

				#select neighbors
        inner_idx = np.where(d_all <= tight_R)[0]
        outer_idx = np.where((d_all > tight_R) & (d_all <= loose_R))[0]

        #fallback if inner region too small
        if len(inner_idx) < 5:
            inner_idx = np.argsort(d_all)[:6]   

        #weighted KNN voting
        w0 = w1 = 0.0

        di = d_all[inner_idx]
        yi = y_train[inner_idx]
        wi = 1.0 / (di**6 + 1e-12)
        w1 += wi[yi == 1].sum()
        w0 += wi[yi == 0].sum()

        #very light outer weight
        if len(outer_idx) > 0:
            do = d_all[outer_idx]
            yo = y_train[outer_idx]
            wo = 0.12 * (1.0 / (do + 1e-12))   # was 0.15
            w1 += wo[yo == 1].sum()
            w0 += wo[yo == 0].sum()

        preds.append(1 if w1 > w0 else 0)

    return np.array(preds, dtype=np.int32)


X_train = np.load("data_train.npy")
y_train = np.load("label_train.npy")

model = train_model(X_train, y_train)

X_test = np.load("data_test.npy")
y_test = run_model(model, X_test).astype(np.int32)

assert y_test.shape == (len(X_test),)
assert y_test.dtype == np.int32

for y in y_test:
    print(y)
