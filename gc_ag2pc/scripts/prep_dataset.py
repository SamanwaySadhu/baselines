#!/usr/bin/env python3
"""Dump a real dataset in the quantized form lr_gc_ag2pc --data reads.

The preprocessing is copied from /ml_training/{iris,wine,titanic,breast_cancer}.py:
same label mapping, same shuffle (random_state=42), same min-max scaling to
[-1, 127/128], same to_int_scaled at bits = 6 (integers in [-32, 31]), and the same
initial weights (uniform in [-1, 1], quantized). The only change is the bias: the .py
files keep a separate random bias; here the all-ones column is appended as the last
feature (the circuit fixes it to MAXQ) and gets its own random initial weight.

usage: prep_dataset.py <iris|wine|titanic|breast_cancer> <ml_training_dir> <out.txt> [seed]
output: line 1 "N D" (D feature columns, no bias); N lines "x_1 ... x_D y";
        last line: the D+1 initial weights (last = bias weight).
"""
import sys
import os
import numpy as np
import pandas as pd

BITS = 6


def to_int_scaled(arr, bits=BITS):
    scaled = np.rint(arr * (2 ** (bits - 1))).astype(np.float64)
    clipped = np.clip(scaled, -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1)
    return clipped.astype(np.int64)


def load(name, root):
    if name == 'iris':
        df = pd.read_csv(os.path.join(root, 'iris.data'), header=None)
        df[4] = (df[4] == 'Iris-setosa').astype(np.int64)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        y = df[4].to_numpy().reshape(-1, 1)
        X_df = df.drop(columns=[4]).astype(float)
    elif name == 'wine':
        df = pd.read_csv(os.path.join(root, 'wine.data'), header=None)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        y = (df[0] == 1).astype(np.int64).to_numpy().reshape(-1, 1)
        X_df = df.drop(columns=[0]).astype(float)
    elif name == 'breast_cancer':
        df = pd.read_csv(os.path.join(root, 'wdbc.data'), header=None)
        df[1] = df[1].map({'B': 0, 'M': 1}).astype(np.int64)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        y = df[1].to_numpy().reshape(-1, 1)
        X_df = df.drop(columns=[0, 1]).astype(float)
    elif name == 'titanic':
        import seaborn as sns
        df = sns.load_dataset('titanic')
        df = df.drop(columns=['class', 'alone', 'alive', 'embark_town', 'who', 'adult_male', 'deck'])
        df['pclass'] = pd.Categorical(df['pclass'], categories=[1, 2, 3])
        pclass_dummies = pd.get_dummies(df['pclass'], prefix='pclass').astype(int)
        df['embarked'] = pd.Categorical(df['embarked'], categories=['C', 'Q', 'S'])
        embarked_dummies = pd.get_dummies(df['embarked'], prefix='embarked').astype(int)
        df = pd.concat([df.drop(columns=['pclass', 'embarked']), pclass_dummies, embarked_dummies], axis=1)
        df['age'] = df['age'].fillna(df['age'].mean())
        df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        y = df['survived'].to_numpy().reshape(-1, 1)
        X_df = df.drop(columns=['survived']).astype(float)
    else:
        raise SystemExit('unknown dataset ' + name)
    return X_df, y.astype(np.int64)


def main():
    name, root, out = sys.argv[1:4]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    X_df, y = load(name, root)
    col_min = X_df.min()
    col_max = X_df.max()
    col_range = (col_max - col_min).replace(0, 1)
    X_df = -1.0 + ((X_df - col_min) / col_range) * (255.0 / 128.0)
    X = to_int_scaled(X_df.to_numpy())
    n, d = X.shape
    np.random.seed(seed)
    w = to_int_scaled(np.random.uniform(-1, 1, size=(d, 1)))      # as in the .py files
    b = to_int_scaled(np.random.uniform(-1, 1, size=(1, 1)))      # becomes the bias-column weight
    w0 = np.concatenate([w.flatten(), b.flatten()])
    with open(out, 'w') as f:
        f.write(f'{n} {d}\n')
        for i in range(n):
            f.write(' '.join(str(int(v)) for v in X[i]) + f' {int(y[i, 0])}\n')
        f.write(' '.join(str(int(v)) for v in w0) + '\n')
    print(f'{name}: N={n} D={d} (+bias => d={d + 1}), positives={int(y.sum())}')


if __name__ == '__main__':
    main()
